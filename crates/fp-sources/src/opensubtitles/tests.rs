use super::*;
use crate::test_server::{TestServer, resp};
use std::sync::atomic::{AtomicUsize, Ordering};

// A test credential made at run time from no string literal, so code
// scanning doesn't take it for a real one.
fn generated() -> String {
    std::process::id().to_string()
}

fn client(server: &TestServer, dir: &tempfile::TempDir) -> Client {
    Client::new(
        &server.base,
        &server.base,
        generated(),
        "FramePlayerX vTest",
        dir.path().join("opensubtitles.json"),
    )
}

#[test]
fn canonical_query_preserves_the_full_language_and_episode_scope() {
    let mut query = SearchQuery {
        title: "A   Café & Its Owner".into(),
        language: Some("PT-BR".into()),
        kind: Some(MediaKind::Episode),
        season: Some(0),
        episode: Some(12),
        parent_imdb_id: Some(99),
        moviehash: Some("AB1234567890CDEF".into()),
        ..Default::default()
    };
    assert_eq!(
        query.canonical(1).unwrap(),
        "ai_translated=exclude&episode_number=12&foreign_parts_only=exclude&languages=pt-br&moviehash=ab1234567890cdef&parent_imdb_id=99&season_number=0&type=episode",
    );
    query.imdb_id = Some(5);
    assert_eq!(
        query.canonical(1).unwrap(),
        "ai_translated=exclude&foreign_parts_only=exclude&imdb_id=5&languages=pt-br&moviehash=ab1234567890cdef&type=episode",
    );
    query.imdb_id = None;
    query.episode = None;
    let text = query.canonical(1).unwrap();
    assert!(text.contains("query=a+caf%C3%A9+%26+its+owner"));
    assert!(!text.contains("imdb"));
    query.moviehash = None;
    query.title = "source://local/private/video.mkv".into();
    assert_eq!(query.canonical(1), Err(Error::InvalidSearch));
    query.title = "/home/user/private/video.mkv".into();
    assert_eq!(query.canonical(1), Err(Error::InvalidSearch));
    query.title = "C:\\Users\\user\\video.mkv".into();
    assert_eq!(query.canonical(1), Err(Error::InvalidSearch));
    query.title = "Face/Off".into();
    assert!(query.canonical(1).unwrap().contains("query=face%2Foff"));
}

#[test]
fn app_languages_map_to_the_services_codes_and_back() {
    for (app, service) in [("pt", "pt-pt"), ("zh", "zh-cn"), ("de", "de")] {
        assert_eq!(language_code(app), service);
        assert_eq!(app_language(service), app);
    }
}

#[test]
fn discovery_and_search_use_cached_responses_and_rank_file_matches() {
    let api = TestServer::start(|req| match req.path() {
        "/infos/languages" => resp(
            200,
            r#"{"data":[{"language_code":"pt-BR","language_name":"Portuguese (Brazilian)"}]}"#,
            &[],
        ),
        "/subtitles" if req.url.contains("page=2") => resp(
            200,
            r#"{"page":2,"total_pages":2,"data":[
                {"attributes":{"language":"pt-br","download_count":900,"files":[{"file_id":50}]}},
                {"attributes":{"language":"pt-br","from_trusted":true,"files":[{"file_id":29,"file_name":"title.srt"}]}}
            ]}"#,
            &[],
        ),
        "/subtitles" => resp(
            200,
            r#"{"page":1,"total_pages":2,"data":[
                {"attributes":{"subtitle_id":"555","language":"pt-br","from_trusted":true,"files":[{"file_id":29,"file_name":"title.srt"}]}},
                {"attributes":{"subtitle_id":"666","language":"pt-br","moviehash_match":true,"files":[{"file_id":44,"file_name":"part1.srt","cd_number":1},{"file_id":45,"file_name":"part2.srt","cd_number":2}]}},
                {"attributes":{"language":"pt","moviehash_match":true,"files":[{"file_id":3}]}},
                {"attributes":{"language":"pt-br","machine_translated":true,"files":[{"file_id":31}]}}
            ]}"#,
            &[],
        ),
        _ => resp(404, "", &[]),
    });
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    let mut query = SearchQuery {
        title: "A Show".into(),
        language: Some("pt-br".into()),
        moviehash: Some("123456789abcdef0".into()),
        ..Default::default()
    };
    assert_eq!(client.languages().unwrap()[0].code, "pt-br");
    assert_eq!(
        client.languages().unwrap()[0].name,
        "Portuguese (Brazilian)"
    );
    // Both pages, ranked together: the file match, then by downloads.
    let results = client.search(&query).unwrap();
    let ids = results.iter().map(|s| s.files[0].id).collect::<Vec<_>>();
    assert_eq!(ids, [44, 50, 29]);
    assert_eq!(results[0].match_kind, MatchKind::File);
    let part = &results[0].files[1];
    assert_eq!(
        (part.id, part.name.as_str(), part.part),
        (45, "part2.srt", Some(2))
    );
    query.title = "A   SHOW".into();
    assert_eq!(client.search(&query).unwrap()[2].files[0].id, 29);
    let requests = api.requests();
    assert_eq!(requests.len(), 3);
    for request in requests {
        assert_eq!(request.header("Api-Key"), Some(generated().as_str()));
        assert_eq!(request.header("User-Agent"), Some("FramePlayerX vTest"));
        assert_eq!(request.header("Authorization"), None);
    }
    assert!(matches!(
        client.search_cancellable(&query, || true),
        Err(Error::Cancelled)
    ));
}

#[test]
fn a_link_retry_sends_no_api_credentials_and_reuses_the_same_file_parameters() {
    let files = TestServer::start(|req| match req.path() {
        "/expired" => resp(410, "", &[]),
        "/redirect" => resp(302, "", &[("Location", "/subtitle?token=short-lived")]),
        "/subtitle" => resp(200, "1\n00:00:00,000 --> 00:00:01,000\nTest\n", &[]),
        _ => resp(404, "", &[]),
    });
    let expired = files.url("/expired");
    let fresh = files.url("/redirect");
    let downloads = AtomicUsize::new(0);
    let api = TestServer::start(move |req| match req.path() {
        "/login" => resp(
            200,
            r#"{"user":{"allowed_downloads":20},"base_url":"api.opensubtitles.com","token":"fixture-token"}"#,
            &[],
        ),
        "/infos/user" => resp(200, r#"{"data":{"remaining_downloads":12}}"#, &[]),
        "/download" => {
            let link = if downloads.fetch_add(1, Ordering::Relaxed) == 0 {
                &expired
            } else {
                &fresh
            };
            resp(200, serde_json::json!({"link":link,"remaining":11,"reset_time_utc":"2099-01-01T00:00:00Z"}).to_string(), &[])
        }
        _ => resp(404, "", &[]),
    });
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    assert_eq!(
        client
            .sign_in(&generated(), &generated())
            .unwrap()
            .remaining,
        Some(12)
    );
    let download = client.download(45).unwrap();
    assert!(download.bytes.ends_with(b"Test\n"));
    assert_eq!(download.quota.remaining, Some(11));
    assert_eq!(
        download.quota.reset_time_utc.as_deref(),
        Some("2099-01-01T00:00:00Z")
    );
    let requests = api.requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(requests[0].header("Authorization"), None);
    for request in &requests[1..] {
        assert_eq!(
            request.header("Authorization"),
            Some("Bearer fixture-token")
        );
    }
    let download_body = serde_json::json!({"file_id":45, "sub_format":"srt"}).to_string();
    assert_eq!(requests[2].body, download_body);
    assert_eq!(requests[3].body, download_body);
    let gets = files.requests();
    assert_eq!(gets.len(), 3);
    for request in gets {
        assert_eq!(request.header("Api-Key"), None);
        assert_eq!(request.header("Authorization"), None);
        assert_eq!(request.method, "GET");
    }
    assert_eq!(client.download(45).unwrap().bytes, download.bytes);
    assert_eq!(api.count(|req| req.path() == "/download"), 2);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(&client.credentials_path).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    }
    client.sign_out().unwrap();
    assert!(!client.signed_in());
    assert!(!client.credentials_path.exists());
    assert_eq!(client.quota().remaining, None);
}

#[test]
fn a_new_link_that_isnt_found_costs_no_second_download() {
    let files = TestServer::start(|_| resp(404, "", &[]));
    let missing = files.url("/missing");
    let api = TestServer::start(move |req| {
        match req.path() {
        "/login" => resp(
            200,
            r#"{"user":{"allowed_downloads":20},"base_url":"api.opensubtitles.com","token":"fixture-token"}"#,
            &[],
        ),
        "/infos/user" => resp(200, r#"{"data":{"remaining_downloads":12}}"#, &[]),
        "/download" => resp(
            200,
            serde_json::json!({"link":missing,"remaining":11,"reset_time_utc":"2099-01-01T00:00:00Z"})
                .to_string(),
            &[],
        ),
        _ => resp(404, "", &[]),
    }
    });
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    client.sign_in(&generated(), &generated()).unwrap();
    assert!(matches!(client.download(45), Err(Error::Unavailable)));
    assert_eq!(api.count(|req| req.path() == "/download"), 1);
}

#[test]
fn reuses_saved_tokens_refreshes_once_and_stops_after_credentials_are_rejected() {
    let logins = AtomicUsize::new(0);
    let api = TestServer::start(move |req| match req.path() {
        "/login" => match logins.fetch_add(1, Ordering::Relaxed) {
            0 => resp(
                200,
                r#"{"user":{},"base_url":"api.opensubtitles.com","token":"old"}"#,
                &[],
            ),
            1 => resp(
                200,
                r#"{"user":{},"base_url":"api.opensubtitles.com","token":"new"}"#,
                &[],
            ),
            _ => resp(401, "", &[]),
        },
        "/infos/user" => resp(200, r#"{"data":{"remaining_downloads":5}}"#, &[]),
        "/download" if req.header("Authorization") == Some("Bearer old") => resp(401, "", &[]),
        "/download" => resp(
            406,
            r#"{"remaining":0,"reset_time_utc":"2099-01-01T00:00:00Z"}"#,
            &[],
        ),
        _ => resp(404, "", &[]),
    });
    let dir = tempfile::tempdir().unwrap();
    client(&api, &dir)
        .sign_in(&generated(), &generated())
        .unwrap();
    let restored = client(&api, &dir);
    assert!(restored.signed_in());
    assert_eq!(restored.user_info().unwrap().remaining, Some(5));
    assert_eq!(api.count(|req| req.path() == "/login"), 1);
    assert!(matches!(
        restored.download(11),
        Err(Error::AllowanceExhausted)
    ));
    assert_eq!(restored.quota().remaining, Some(0));
    let attempts = api
        .requests()
        .into_iter()
        .filter(|req| req.path() == "/download")
        .collect::<Vec<_>>();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].body, attempts[1].body);
    assert_eq!(attempts[1].header("Authorization"), Some("Bearer new"));
    let mut account = load_account(&restored.credentials_path).unwrap().unwrap();
    account.issued_at -= TOKEN_TTL_SECS;
    restored.save_account(&account).unwrap();
    for _ in 0..2 {
        assert!(matches!(
            client(&api, &dir).download(11),
            Err(Error::CredentialsRejected)
        ));
    }
    assert_eq!(api.count(|req| req.path() == "/login"), 3);
    assert_eq!(api.count(|req| req.path() == "/download"), 2);
}

#[test]
fn restricts_api_hosts_and_honors_rate_limits_without_replaying_a_post() {
    assert_eq!(api_base("api.opensubtitles.com").unwrap(), API);
    assert_eq!(api_base("vip-api.opensubtitles.com").unwrap(), VIP_API);
    for host in [
        "api.opensubtitles.com.invalid",
        "api.opensubtitles.com@invalid",
        "https://api.opensubtitles.com",
    ] {
        assert_eq!(api_base(host), Err(Error::InvalidResponse));
    }
    let api = TestServer::start(|_| resp(429, "rate limit", &[("Retry-After", "60")]));
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    assert!(
        matches!(client.download(10), Err(Error::RateLimited { retry_after }) if retry_after == Duration::from_secs(60))
    );
    assert!(
        matches!(client.download(10), Err(Error::RateLimited { retry_after }) if retry_after > Duration::from_secs(50))
    );
    assert_eq!(api.requests().len(), 1);
}

#[test]
fn rejects_oversized_or_non_utf8_downloads() {
    let files = TestServer::start(|req| match req.path() {
        "/huge" => resp(200, vec![b'a'; SIZE_LIMIT as usize + 1], &[]),
        _ => resp(200, vec![0xff, 0xfe], &[]),
    });
    let huge = files.url("/huge");
    let invalid = files.url("/invalid");
    let api = TestServer::start(move |req| {
        let data = serde_json::from_str::<serde_json::Value>(&req.body).unwrap();
        let link = if data["file_id"] == 1 {
            &huge
        } else {
            &invalid
        };
        resp(200, serde_json::json!({"link":link}).to_string(), &[])
    });
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    assert!(matches!(client.download(1), Err(Error::InvalidSubtitle)));
    assert!(matches!(client.download(2), Err(Error::InvalidSubtitle)));
}

#[test]
fn vip_api_receives_the_token_but_cross_origin_redirects_receive_nothing() {
    let foreign = TestServer::start(|_| resp(500, "", &[]));
    let destination = foreign.url("/surprise");
    let vip = TestServer::start(move |req| match req.path() {
        "/infos/user" => resp(200, r#"{"data":{"remaining_downloads":9}}"#, &[]),
        "/infos/languages" => resp(
            200,
            r#"{"data":[{"language_code":"en","language_name":"English"}]}"#,
            &[],
        ),
        "/subtitles" => resp(307, "", &[("Location", &destination)]),
        _ => resp(404, "", &[]),
    });
    let api = TestServer::start(|req| match req.path() {
        "/login" => resp(
            200,
            r#"{"user":{},"base_url":"vip-api.opensubtitles.com","token":"vip-token"}"#,
            &[],
        ),
        _ => resp(404, "", &[]),
    });
    let dir = tempfile::tempdir().unwrap();
    let client = Client::new(
        &api.base,
        &vip.base,
        generated(),
        "FramePlayerX vTest",
        dir.path().join("opensubtitles.json"),
    );
    assert_eq!(
        client
            .sign_in(&generated(), &generated())
            .unwrap()
            .remaining,
        Some(9)
    );
    assert_eq!(client.languages().unwrap()[0].name, "English");
    let query = SearchQuery {
        title: "A movie".into(),
        ..Default::default()
    };
    assert!(matches!(client.search(&query), Err(Error::InvalidResponse)));
    assert_eq!(api.requests().len(), 1);
    assert_eq!(vip.count(|req| req.path() == "/subtitles"), 1);
    for req in vip.requests() {
        assert_eq!(req.header("Authorization"), Some("Bearer vip-token"));
        assert_eq!(req.header("Api-Key"), Some(generated().as_str()));
    }
    assert!(foreign.requests().is_empty());
}

#[test]
fn a_file_name_is_searched_in_place_of_the_title_and_year() {
    let mut query = SearchQuery {
        title: "Blade Runner 2049".into(),
        year: Some(2017),
        file_name: Some("Blade Runner 2049 (2017) [1080p].mkv".into()),
        ..Default::default()
    };
    assert_eq!(
        query.canonical(1).unwrap(),
        "ai_translated=exclude&foreign_parts_only=exclude&query=blade+runner+2049+%282017%29+%5B1080p%5D.mkv",
    );
    assert!(query.canonical(2).unwrap().contains("&page=2&"));
    query.file_name = None;
    assert!(
        query
            .canonical(1)
            .unwrap()
            .ends_with("&query=blade+runner+2049&year=2017")
    );
    query.file_name = Some("Blade Runner 2049 (2017) [1080p].mkv".into());
    query.imdb_id = Some(1856101);
    assert!(!query.canonical(1).unwrap().contains("query="));
}

#[test]
fn keeps_only_the_same_episode_or_movie_and_puts_file_matches_then_downloads_first() {
    let api = TestServer::start(|req| {
        let body = if req.url.contains("imdb_id=133093") {
            r#"{"page":1,"total_pages":1,"data":[
                {"attributes":{"language":"en","download_count":900,"feature_details":{"feature_type":"Movie","imdb_id":999},"files":[{"file_id":7}]}},
                {"attributes":{"language":"en","download_count":1000,"feature_details":{"feature_type":"Episode"},"files":[{"file_id":8}]}},
                {"attributes":{"language":"en","download_count":5,"feature_details":{"feature_type":"Movie","imdb_id":133093},"files":[{"file_id":6}]}}
            ]}"#
        } else {
            r#"{"page":1,"total_pages":1,"data":[
                {"attributes":{"language":"en","download_count":900,"feature_details":{"feature_type":"Episode","season_number":2,"episode_number":2},"files":[{"file_id":1}]}},
                {"attributes":{"language":"en","download_count":800,"feature_details":{"feature_type":"Movie"},"files":[{"file_id":2}]}},
                {"attributes":{"language":"en","download_count":10,"feature_details":{"feature_type":"Episode","season_number":2,"episode_number":1},"files":[{"file_id":3}]}},
                {"attributes":{"language":"en","download_count":500,"feature_details":{"feature_type":"Episode","season_number":"2","episode_number":"1"},"files":[{"file_id":4}]}},
                {"attributes":{"language":"en","download_count":1,"moviehash_match":true,"feature_details":{"feature_type":"Episode","season_number":2,"episode_number":1},"files":[{"file_id":5}]}}
            ]}"#
        };
        resp(200, body, &[])
    });
    let dir = tempfile::tempdir().unwrap();
    let client = client(&api, &dir);
    let ids = |results: Vec<Subtitle>| results.iter().map(|s| s.files[0].id).collect::<Vec<_>>();
    let episode = SearchQuery {
        kind: Some(MediaKind::Episode),
        season: Some(2),
        episode: Some(1),
        file_name: Some("The.Office.US.S02E01.720p.mkv".into()),
        moviehash: Some("123456789abcdef0".into()),
        ..Default::default()
    };
    assert_eq!(ids(client.search(&episode).unwrap()), [5, 4, 3]);
    let movie = SearchQuery {
        kind: Some(MediaKind::Movie),
        imdb_id: Some(133093),
        ..Default::default()
    };
    assert_eq!(ids(client.search(&movie).unwrap()), [6]);
}
