//! Integration tests for the Jellyfin/Emby MediaBrowser source.

mod common;

use common::{Req, TestServer, pattern, resp, serve_bytes};
use fp_core::format::{Projection, StereoLayout};
use fp_core::source::EntryKind;
use fp_sources::{
    Credentials, MediaBrowserConfig, SourceConfig, SourceKind, WatchEvent, WatchKind, build,
};
use std::sync::Arc;

fn mediabrowser_config(url: String, creds: Option<Credentials>) -> MediaBrowserConfig {
    MediaBrowserConfig {
        id: "mb-test".into(),
        name: "Test Jellyfin".into(),
        url,
        credentials: creds,
        insecure_tls: false,
        max_height: None,
        access_token: None,
        user_id: None,
    }
}

#[test]
fn mediabrowser_full_lifecycle_browse_stream_and_progress() {
    let video_bytes = Arc::new(pattern(100_000));
    let v_bytes = video_bytes.clone();

    let server = TestServer::start(move |req: &Req| {
        let auth = req.header("authorization").unwrap_or("");

        // Public info (unauthed)
        if req.path() == "/System/Info/Public" {
            return resp(
                200,
                r#"{"ProductName":"Jellyfin Server","Version":"10.9.11"}"#,
                &[("Content-Type", "application/json")],
            );
        }

        // Login
        if req.path() == "/Users/AuthenticateByName" {
            assert!(auth.contains("MediaBrowser Client=\"FramePlayerX\""));
            assert!(req.body.contains("alice") && req.body.contains("secret123"));
            return resp(
                200,
                r#"{"AccessToken":"test_token_abc","User":{"Id":"usr_42","Name":"alice"}}"#,
                &[("Content-Type", "application/json")],
            );
        }

        // Must be authenticated for subsequent calls
        assert!(auth.contains("Token=\"test_token_abc\""));

        match req.path() {
            // Views (libraries)
            "/Users/usr_42/Views" => resp(
                200,
                r#"{"Items":[{"Id":"view_movies","Name":"Movies","Type":"CollectionFolder"},{"Id":"view_vr","Name":"VR Videos","Type":"CollectionFolder"}],"TotalRecordCount":2}"#,
                &[("Content-Type", "application/json")],
            ),
            // Items under view_vr
            "/Users/usr_42/Items" => {
                assert!(req.url.contains("ParentId=view_vr"));
                resp(
                    200,
                    r#"{"Items":[
                        {"Id":"folder_trips","Name":"Trips","Type":"Folder"},
                        {"Id":"item_rollercoaster","Name":"Rollercoaster","Type":"Movie","RunTimeTicks":600000000,"Path":"/media/VR/Rollercoaster_180_LR.mp4"}
                    ],"TotalRecordCount":2}"#,
                    &[("Content-Type", "application/json")],
                )
            }
            // Item details
            "/Users/usr_42/Items/item_rollercoaster" => resp(
                200,
                r#"{
                    "Id":"item_rollercoaster",
                    "Name":"Rollercoaster",
                    "Type":"Movie",
                    "RunTimeTicks":600000000,
                    "Path":"/media/VR/Rollercoaster_180_LR.mp4",
                    "UserData":{"PlaybackPositionTicks":123000000,"Played":false},
                    "MediaSources":[{
                        "Id":"ms_1",
                        "MediaStreams":[
                            {"Index":0,"Type":"Video","Width":3840,"Height":1920,"Codec":"hevc"},
                            {"Index":1,"Type":"Audio","Codec":"aac"},
                            {"Index":2,"Type":"Subtitle","Codec":"srt","Language":"eng","IsExternal":true}
                        ]
                    }],
                    "Chapters":[
                        {"StartPositionTicks":0,"Name":"Intro"},
                        {"StartPositionTicks":300000000,"Name":"Drop"}
                    ]
                }"#,
                &[("Content-Type", "application/json")],
            ),
            // Direct video streaming
            "/Videos/item_rollercoaster/stream" => serve_bytes(req, &v_bytes, true),
            // Playback session reporting
            "/Sessions/Playing" | "/Sessions/Playing/Progress" | "/Sessions/Playing/Stopped" => {
                resp(204, "", &[])
            }
            _ => resp(404, "", &[]),
        }
    });

    let config = mediabrowser_config(server.url(""), Some(Credentials::new("alice", "secret123")));
    let src = build(&SourceConfig::MediaBrowser(config)).unwrap();
    assert_eq!(src.kind(), SourceKind::MediaBrowser);

    // 1. List views (root)
    let views = src.list(None).unwrap();
    assert_eq!(views.len(), 2);
    assert_eq!(views[0].name, "Movies");
    assert_eq!(views[0].kind, EntryKind::Directory);
    assert_eq!(views[1].name, "VR Videos");
    assert_eq!(views[1].kind, EntryKind::Directory);

    // 2. List items in VR Videos
    let vr_items = src.list(Some(&views[1].location)).unwrap();
    assert_eq!(vr_items.len(), 2);
    assert_eq!(vr_items[0].name, "Trips");
    assert_eq!(vr_items[0].kind, EntryKind::Directory);
    assert_eq!(vr_items[1].name, "Rollercoaster");
    assert_eq!(vr_items[1].kind, EntryKind::Video);
    assert_eq!(vr_items[1].duration, Some(60.0));

    // VR format detected from path
    let fmt = vr_items[1].format.unwrap();
    assert_eq!(fmt.projection, Projection::EQUIRECT_180);
    assert_eq!(fmt.stereo, StereoLayout::SideBySide);

    // 3. Get details
    let detailed = src.details(&vr_items[1]).unwrap();
    assert_eq!(detailed.markers.len(), 2);
    assert_eq!(detailed.markers[0], (0.0, "Intro".to_string()));
    assert_eq!(detailed.markers[1], (30.0, "Drop".to_string()));
    assert_eq!(detailed.subtitles.len(), 1);
    assert!(
        detailed.subtitles[0]
            .location
            .contains("/Videos/item_rollercoaster/ms_1/Subtitles/2/Stream.srt")
    );

    // 4. Sidecars
    let sidecars = src.sidecars(&vr_items[1]).unwrap();
    assert_eq!(sidecars.subtitles.len(), 1);
    assert_eq!(sidecars.subtitles[0].language.as_deref(), Some("eng"));

    // 5. Open stream and random access read
    let f = src.open(&vr_items[1].location).unwrap();
    assert_eq!(f.size(), Some(100_000));
    let mut buf = [0u8; 100];
    assert_eq!(f.read_at(50_000, &mut buf).unwrap(), 100);
    assert_eq!(buf, video_bytes[50_000..50_100]);

    // 6. Watch sync: one session's reports, in the server's terms
    assert_eq!(detailed.resume, Some(12.3));
    let session = src.watch_session(&detailed).unwrap();
    assert_eq!(session.item, "item_rollercoaster");
    for (kind, position, paused) in [
        (WatchKind::Start, 0.0, false),
        (WatchKind::Tick, 10.0, false),
        (WatchKind::Pause, 15.0, true),
        (WatchKind::Resume, 15.0, false),
        (WatchKind::Seek, 40.0, false),
        (WatchKind::Stop, 60.0, false),
    ] {
        let event = WatchEvent {
            kind,
            position,
            paused,
            speed: 1.0,
        };
        src.report_watch(&session, event).unwrap();
    }
    let reports: Vec<serde_json::Value> = server
        .requests()
        .iter()
        .filter(|r| r.path().starts_with("/Sessions/Playing"))
        .map(|r| {
            let mut v: serde_json::Value = serde_json::from_str(&r.body).unwrap();
            v["path"] = r.path().into();
            v
        })
        .collect();
    let summary: Vec<_> = reports
        .iter()
        .map(|v| {
            (
                v["path"].as_str().unwrap(),
                v["EventName"].as_str().unwrap_or(""),
                v["PositionTicks"].as_i64().unwrap(),
                v["IsPaused"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("/Sessions/Playing", "", 0, false),
            (
                "/Sessions/Playing/Progress",
                "TimeUpdate",
                100_000_000,
                false
            ),
            ("/Sessions/Playing/Progress", "Pause", 150_000_000, true),
            ("/Sessions/Playing/Progress", "Unpause", 150_000_000, false),
            (
                "/Sessions/Playing/Progress",
                "TimeUpdate",
                400_000_000,
                false
            ),
            ("/Sessions/Playing/Stopped", "", 600_000_000, false),
        ]
    );
    assert!(
        reports
            .iter()
            .all(|v| v["ItemId"] == "item_rollercoaster"
                && v["PlaySessionId"] == session.id.as_str())
    );
}

#[test]
fn mediabrowser_search_finds_titles_by_kind_not_episodes() {
    let json = [("Content-Type", "application/json")];
    let server = TestServer::start(move |req: &Req| match req.path() {
        "/Users/AuthenticateByName" => resp(
            200,
            r#"{"AccessToken":"t","User":{"Id":"u1","Name":"alice"}}"#,
            &json,
        ),
        "/Users/u1/Items" => {
            // One request per section, with no order: the server ranks.
            assert!(req.url.contains("SearchTerm=alien&"), "{}", req.url);
            assert!(!req.url.contains("SortBy"), "{}", req.url);
            assert!(!req.url.contains("Episode"), "{}", req.url);
            let body = if req.url.contains("IncludeItemTypes=Movie&") {
                r#"{"Items":[{"Id":"m1","Name":"Alien","Type":"Movie"},
                             {"Id":"m2","Name":"Aliens","Type":"Movie"}]}"#
            } else if req.url.contains("IncludeItemTypes=Series&") {
                r#"{"Items":[{"Id":"s1","Name":"Alien: Earth","Type":"Series"}]}"#
            } else {
                r#"{"Items":[]}"#
            };
            resp(200, body, &json)
        }
        _ => resp(404, "", &[]),
    });
    let mb = fp_sources::MediaBrowserSource::new(&mediabrowser_config(
        server.url(""),
        Some(Credentials::new("alice", "secret123")),
    ))
    .unwrap();
    let found = mb.search(" alien ", 50).unwrap();
    let sections: Vec<(&str, Vec<&str>)> = found
        .iter()
        .map(|(s, items)| (*s, items.iter().map(|i| i.name.as_str()).collect()))
        .collect();
    assert_eq!(
        sections,
        [
            ("Movies", vec!["Alien", "Aliens"]),
            ("Shows", vec!["Alien: Earth"])
        ]
    );
}

#[test]
fn mediabrowser_quick_connect_flow() {
    let server = TestServer::start(|req: &Req| match req.path() {
        "/QuickConnect/Enabled" => resp(200, "true", &[("Content-Type", "application/json")]),
        "/QuickConnect/Initiate" => resp(
            200,
            r#"{"Code":"849201","Secret":"sec_xyz123"}"#,
            &[("Content-Type", "application/json")],
        ),
        "/QuickConnect/Connect" => {
            if req.url.contains("sec_xyz123") {
                resp(
                    200,
                    r#"{"Authenticated":true,"Code":"849201","Secret":"sec_xyz123"}"#,
                    &[("Content-Type", "application/json")],
                )
            } else {
                resp(
                    200,
                    r#"{"Authenticated":false}"#,
                    &[("Content-Type", "application/json")],
                )
            }
        }
        "/Users/AuthenticateWithQuickConnect" => {
            assert!(req.body.contains("sec_xyz123"));
            resp(
                200,
                r#"{"AccessToken":"qc_token_99","User":{"Id":"usr_bob"}}"#,
                &[("Content-Type", "application/json")],
            )
        }
        _ => resp(404, "", &[]),
    });

    let config = mediabrowser_config(server.url(""), None);
    let mb = fp_sources::MediaBrowserSource::new(&config).unwrap();

    assert!(mb.quick_connect_enabled().unwrap());
    let init = mb.quick_connect_initiate().unwrap();
    assert_eq!(init.code, "849201");
    assert_eq!(init.secret, "sec_xyz123");

    let authed = mb.quick_connect_poll(&init.secret).unwrap();
    assert!(authed);

    let (token, uid) = mb.auth_credentials().unwrap();
    assert_eq!(token, "qc_token_99");
    assert_eq!(uid, "usr_bob");
}
