//! OpenSubtitles.com search, account and download requests.

mod query;
pub use query::{
    ContentFilter, Language, MatchKind, MediaKind, SdhFilter, SearchQuery, Subtitle, SubtitleFile,
};

pub use crate::config::Credentials;
use crate::config::write_private;
use crate::http::{HttpClient, HttpOptions, HttpReply};
use crate::timeutil::{parse_http_date, system_time_to_unix};
use fp_core::sync::lock;
use query::SearchReply;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

const API: &str = "https://api.opensubtitles.com/api/v1";
const VIP_API: &str = "https://vip-api.opensubtitles.com/api/v1";
const SIZE_LIMIT: u64 = 8 * 1024 * 1024;
const TOKEN_TTL_SECS: i64 = 24 * 60 * 60;
const SEARCH_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
// 500 results; a movie in one language rarely has more than two pages.
const MAX_PAGES: u32 = 10;
const FILE_LINK_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_RETRY_WAIT: Duration = Duration::from_secs(2);
const REQUEST_SPACING: Duration = Duration::from_millis(250);

/// Why a download link gave no file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinkFailed {
    /// The server says it has run out.
    Expired,
    /// Refused or not found.
    Refused,
}
const SUB_FORMAT: &str = "srt";

/// Identity of the conversion parameters saved with an attachment.
pub const DOWNLOAD_PARAMETERS: &str = "srt:utf-8";

/// The service's code for a language `fp_core::language` names: it tells
/// Portuguese and Chinese apart by where they are written.
pub fn language_code(code: &str) -> &str {
    match code {
        "pt" => "pt-pt",
        "zh" => "zh-cn",
        code => code,
    }
}

/// The `fp_core::language` code for one of the service's, the reverse of
/// [`language_code`].
pub fn app_language(code: &str) -> &str {
    match code {
        "pt-pt" => "pt",
        "zh-cn" => "zh",
        code => code,
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// The errors contain no provider messages, credentials, or download links.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("OpenSubtitles is not configured in this build")]
    Unconfigured,
    #[error("Enter a video title to search for subtitles")]
    InvalidSearch,
    #[error("This subtitle language is unavailable")]
    UnavailableLanguage,
    #[error("OpenSubtitles rejected the sign-in")]
    CredentialsRejected,
    #[error("Sign in to OpenSubtitles to download subtitles")]
    AccountRequired,
    #[error("The OpenSubtitles download allowance is exhausted")]
    AllowanceExhausted,
    #[error("OpenSubtitles has limited requests temporarily")]
    RateLimited { retry_after: Duration },
    #[error("Could not reach OpenSubtitles")]
    Unavailable,
    #[error("OpenSubtitles returned an invalid response")]
    InvalidResponse,
    #[error("The subtitle file was invalid or too large")]
    InvalidSubtitle,
    #[error("Could not save the OpenSubtitles sign-in")]
    Storage,
    #[error("OpenSubtitles search was cancelled")]
    Cancelled,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Quota {
    pub remaining: Option<u32>,
    pub reset_time_utc: Option<String>,
    pub allowed: Option<u32>,
}

impl Quota {
    fn update_reset_time(&mut self, reset_time: Option<String>) {
        if let Some(reset) = reset_time
            && crate::timeutil::parse_iso8601(&reset).is_some()
        {
            self.reset_time_utc = Some(reset);
        }
    }
}

pub struct Download {
    pub bytes: Vec<u8>,
    pub quota: Quota,
}

#[derive(Clone, Deserialize, Serialize)]
struct Account {
    credentials: Credentials,
    token: String,
    api_host: String,
    issued_at: i64,
    #[serde(default)]
    rejected: bool,
}

impl Account {
    fn token_expired(&self) -> bool {
        let elapsed = unix_now().saturating_sub(self.issued_at);
        !(0..TOKEN_TTL_SECS).contains(&elapsed)
    }
}

#[derive(Default)]
struct AccountStatus {
    saved: Option<Account>,
    quota: Quota,
}

#[derive(Default)]
struct Requests {
    next_api: Option<Instant>,
    next_file: Option<Instant>,
    next_login: Option<Instant>,
    searches: VecDeque<(String, Instant, SearchReply)>,
    languages: Option<Vec<Language>>,
    links: HashMap<u64, FileLink>,
}

struct FileLink {
    url: url::Url,
    until: Instant,
}

/// One client per application. Network operations block and belong on worker
/// threads. Account and quota reads do not wait for network operations.
pub struct Client {
    api: HttpClient,
    vip_api: HttpClient,
    files: HttpClient,
    default_base: String,
    vip_base: String,
    available: bool,
    credentials_path: PathBuf,
    status: Mutex<AccountStatus>,
    requests: Mutex<Requests>,
}

impl Client {
    pub fn configured(user_agent: &str, credentials_path: PathBuf) -> Self {
        let mut key = option_env!("FP_OPENSUBTITLES_API_KEY")
            .unwrap_or_default()
            .to_owned();
        if cfg!(debug_assertions)
            && let Ok(override_key) = std::env::var("FP_OPENSUBTITLES_API_KEY")
            && !override_key.trim().is_empty()
        {
            key = override_key;
        }
        Self::new(
            API,
            VIP_API,
            key.trim().to_owned(),
            user_agent,
            credentials_path,
        )
    }

    fn new(
        base: &str,
        vip_base: &str,
        key: String,
        user_agent: &str,
        credentials_path: PathBuf,
    ) -> Self {
        let options = HttpOptions {
            user_agent: user_agent.into(),
            // Retries for requests that consume quota are driven here.
            max_retries: 0,
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(8),
            ..HttpOptions::default()
        };
        let api_client = |base| {
            HttpClient::new_with_headers(
                options.clone(),
                vec![("Api-Key".into(), key.clone())],
                base,
            )
        };
        let api = api_client(base);
        let vip_api = api_client(vip_base);
        let files = HttpClient::new(options, None).without_redirects();
        let status = AccountStatus {
            saved: load_account(&credentials_path).unwrap_or_else(|e| {
                log::warn!("cannot restore OpenSubtitles sign-in: {e}");
                None
            }),
            quota: Quota::default(),
        };
        Self {
            api,
            vip_api,
            files,
            default_base: base.to_owned(),
            vip_base: vip_base.to_owned(),
            available: !key.is_empty(),
            credentials_path,
            status: Mutex::new(status),
            requests: Mutex::new(Requests::default()),
        }
    }

    pub fn available(&self) -> bool {
        self.available
    }

    /// Whether an account is saved on this device.
    pub fn signed_in(&self) -> bool {
        lock(&self.status).saved.is_some()
    }

    pub fn account_rejected(&self) -> bool {
        lock(&self.status)
            .saved
            .as_ref()
            .is_some_and(|account| account.rejected)
    }

    pub fn quota(&self) -> Quota {
        let quota = lock(&self.status).quota.clone();
        if quota
            .reset_time_utc
            .as_deref()
            .and_then(crate::timeutil::parse_iso8601)
            .is_some_and(|reset| reset <= unix_now())
        {
            return Quota {
                allowed: quota.allowed,
                ..Quota::default()
            };
        }
        quota
    }

    pub fn sign_in(&self, username: &str, password: &str) -> Result<Quota> {
        let mut requests = lock(&self.requests);
        let credentials = Credentials::new(username.trim(), password);
        if credentials.username.is_empty() || credentials.password.is_empty() {
            return Err(Error::CredentialsRejected);
        }
        let (account, quota) = self.login(&mut requests, credentials, &|| false)?;
        self.save_account(&account)?;
        requests.links.clear();
        requests.searches.clear();
        *lock(&self.status) = AccountStatus {
            saved: Some(account),
            quota,
        };
        // Login reports the account tier's limit, while user info reports
        // what remains. Preserve the sign-in if only the latter fails.
        if matches!(
            self.refresh_user_info(&mut requests, &|| false),
            Err(Error::CredentialsRejected)
        ) {
            return Err(Error::CredentialsRejected);
        }
        Ok(self.quota())
    }

    pub fn sign_out(&self) -> Result<()> {
        let mut requests = lock(&self.requests);
        let mut temp_name = self.credentials_path.as_os_str().to_owned();
        temp_name.push(".tmp");
        let temp_path = PathBuf::from(temp_name);
        for path in [&temp_path, &self.credentials_path] {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err(Error::Storage),
            }
        }
        requests.links.clear();
        requests.searches.clear();
        *lock(&self.status) = AccountStatus::default();
        Ok(())
    }

    pub fn user_info(&self) -> Result<Quota> {
        self.user_info_cancellable(|| false)
    }

    pub fn user_info_cancellable(&self, cancelled: impl Fn() -> bool) -> Result<Quota> {
        self.refresh_user_info(&mut lock(&self.requests), &cancelled)
    }

    fn refresh_user_info(
        &self,
        requests: &mut Requests,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Quota> {
        if !self.signed_in() {
            return Err(Error::AccountRequired);
        }
        let reply =
            self.account_request_cancellable(requests, ("GET", "/infos/user", None), cancelled)?;
        check_cancelled(cancelled)?;
        let info = json::<Data<UserInfo>>(&reply, |_| Error::InvalidResponse)?.data;
        let mut status = lock(&self.status);
        status.quota.allowed = info.allowed_downloads.or(status.quota.allowed);
        status.quota.remaining = info.remaining_downloads;
        status.quota.update_reset_time(info.reset_time_utc);
        Ok(status.quota.clone())
    }

    pub fn languages(&self) -> Result<Vec<Language>> {
        self.languages_cancellable(|| false)
    }

    pub fn languages_cancellable(&self, cancelled: impl Fn() -> bool) -> Result<Vec<Language>> {
        let mut requests = lock(&self.requests);
        check_cancelled(&cancelled)?;
        if let Some(languages) = requests.languages.as_ref() {
            return Ok(languages.clone());
        }
        let reply = self.account_request_cancellable(
            &mut requests,
            ("GET", "/infos/languages", None),
            &cancelled,
        )?;
        check_cancelled(&cancelled)?;
        let mut languages = json::<Data<Vec<Language>>>(&reply, |_| Error::InvalidResponse)?.data;
        languages.retain(|language| !language.code.is_empty() && !language.name.is_empty());
        // Searches send codes in lower case, and the app compares them so.
        for language in &mut languages {
            language.code.make_ascii_lowercase();
        }
        if languages.is_empty() {
            return Err(Error::InvalidResponse);
        }
        languages.sort_by_cached_key(|language| language.name.to_lowercase());
        requests.languages = Some(languages.clone());
        Ok(languages)
    }

    pub fn search(&self, query: &SearchQuery) -> Result<Vec<Subtitle>> {
        self.search_cancellable(query, || false)
    }

    /// Every page of results, ranked together.
    pub fn search_cancellable(
        &self,
        query: &SearchQuery,
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<Subtitle>> {
        query.canonical(1)?;
        let mut requests = lock(&self.requests);
        check_cancelled(&cancelled)?;
        if let (Some(code), Some(languages)) = (&query.language, &requests.languages)
            && !code.trim().is_empty()
            && !languages
                .iter()
                .any(|language| language.code.eq_ignore_ascii_case(code.trim()))
        {
            return Err(Error::UnavailableLanguage);
        }
        let mut results: Option<SearchReply> = None;
        for page in 1..=MAX_PAGES {
            let reply = self.search_page(&mut requests, &query.canonical(page)?, &cancelled)?;
            let last = reply.last_page();
            match &mut results {
                Some(results) => results.append(reply),
                None => results = Some(reply),
            }
            if page >= last {
                break;
            }
        }
        Ok(results.map_or_else(Vec::new, |results| results.rank(query)))
    }

    fn search_page(
        &self,
        requests: &mut Requests,
        canonical: &str,
        cancelled: &impl Fn() -> bool,
    ) -> Result<SearchReply> {
        if let Some((_, _, cached)) = requests
            .searches
            .iter()
            .find(|(key, since, _)| *key == canonical && since.elapsed() < SEARCH_CACHE_TTL)
        {
            return Ok(cached.clone());
        }
        let reply = self.account_request_cancellable(
            requests,
            ("GET", &format!("/subtitles?{canonical}"), None),
            cancelled,
        )?;
        check_cancelled(cancelled)?;
        let result = json::<SearchReply>(&reply, |status| match status {
            400 | 422 => Error::InvalidSearch,
            _ => Error::InvalidResponse,
        })?;
        requests
            .searches
            .retain(|(_, since, _)| since.elapsed() < SEARCH_CACHE_TTL);
        while requests.searches.len() >= 4 * MAX_PAGES as usize {
            requests.searches.pop_back();
        }
        requests
            .searches
            .push_front((canonical.to_owned(), Instant::now(), result.clone()));
        Ok(result)
    }

    pub fn download(&self, file_id: u64) -> Result<Download> {
        if file_id == 0 {
            return Err(Error::InvalidSubtitle);
        }
        let mut requests = lock(&self.requests);
        requests.links.retain(|_, link| link.until > Instant::now());
        let cached = requests.links.get(&file_id).map(|link| link.url.clone());
        let from_cache = cached.is_some();
        let link = match cached {
            Some(url) => url,
            None => self.request_file(&mut requests, file_id)?,
        };
        let bytes = match self.fetch_file(&mut requests, link)? {
            Ok(body) => body,
            // A link kept from before may have run out, or the server says
            // this one has: request the exact same file and parameters once
            // more. A new link refused otherwise won't do better asked for
            // again, and each one counts against the day's downloads.
            // Retryable GET failures keep the link.
            Err(why) if from_cache || why == LinkFailed::Expired => {
                requests.links.remove(&file_id);
                let link = self.request_file(&mut requests, file_id)?;
                match self.fetch_file(&mut requests, link)? {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        requests.links.remove(&file_id);
                        return Err(Error::Unavailable);
                    }
                }
            }
            Err(_) => {
                requests.links.remove(&file_id);
                return Err(Error::Unavailable);
            }
        };
        Ok(Download {
            bytes,
            quota: self.quota(),
        })
    }

    fn request_file(&self, requests: &mut Requests, file_id: u64) -> Result<url::Url> {
        let body = serde_json::json!({ "file_id": file_id, "sub_format": SUB_FORMAT }).to_string();
        let reply = self.account_request(requests, "POST", "/download", Some(body.as_bytes()))?;
        let link = serde_json::from_slice::<LinkReply>(&reply.body);
        if reply.status == 406
            || (reply.status == 429
                && link.as_ref().is_ok_and(|data| {
                    data.remaining == Some(0)
                        || data
                            .reset_time_utc
                            .as_ref()
                            .is_some_and(|time| !time.is_empty())
                }))
        {
            if reply.header("retry-after").is_none() && reply.header("ratelimit-reset").is_none() {
                // Account allowance and the API request window are independent.
                requests.next_api = Some(Instant::now() + REQUEST_SPACING);
            }
            self.update_download_quota(Some(0), link.ok().and_then(|data| data.reset_time_utc));
            return Err(Error::AllowanceExhausted);
        }
        response_ok(&reply, |status| match status {
            401 | 403 => Error::AccountRequired,
            404 | 410 => Error::InvalidSubtitle,
            _ => Error::InvalidResponse,
        })?;
        let link = link.map_err(|_| Error::InvalidResponse)?;
        self.update_download_quota(link.remaining, link.reset_time_utc);
        let url = url::Url::parse(link.link.as_deref().ok_or(Error::InvalidResponse)?)
            .map_err(|_| Error::InvalidResponse)?;
        self.check_file_url(&url)?;
        // Hold links briefly in memory so a failed transfer or save can retry
        // via GET without another allowance-consuming request.
        if requests.links.len() >= 16 {
            requests.links.clear();
        }
        requests.links.insert(
            file_id,
            FileLink {
                url: url.clone(),
                until: Instant::now() + FILE_LINK_TTL,
            },
        );
        Ok(url)
    }

    fn update_download_quota(&self, remaining: Option<u32>, reset_time_utc: Option<String>) {
        let mut status = lock(&self.status);
        status.quota.remaining = remaining;
        status.quota.update_reset_time(reset_time_utc);
    }

    fn fetch_file(
        &self,
        requests: &mut Requests,
        mut url: url::Url,
    ) -> Result<std::result::Result<Vec<u8>, LinkFailed>> {
        let mut retried = false;
        for _ in 0..10 {
            self.check_file_url(&url)?;
            wait_until(requests.next_file, &|| false)?;
            let reply =
                self.files
                    .request_no_status_retry("GET", url.as_str(), &[], None, SIZE_LIMIT + 1);
            let Ok(reply) = reply else {
                if retried {
                    return Err(Error::Unavailable);
                }
                retried = true;
                continue;
            };
            let delay = update_window(&mut requests.next_file, &reply);
            match reply.status {
                200 => {
                    if reply.body.is_empty()
                        || reply.body.len() as u64 > SIZE_LIMIT
                        || std::str::from_utf8(&reply.body).is_err()
                    {
                        return Err(Error::InvalidSubtitle);
                    }
                    return Ok(Ok(reply.body));
                }
                301 | 302 | 303 | 307 | 308 => {
                    let location = reply.header("location").ok_or(Error::InvalidResponse)?;
                    url = url.join(location).map_err(|_| Error::InvalidResponse)?;
                }
                410 => return Ok(Err(LinkFailed::Expired)),
                401 | 403 | 404 => return Ok(Err(LinkFailed::Refused)),
                408 | 429 | 500 | 502 | 503 | 504 if !retried && delay <= MAX_RETRY_WAIT => {
                    retried = true;
                }
                429 => return Err(Error::RateLimited { retry_after: delay }),
                _ => return Err(Error::Unavailable),
            }
        }
        Err(Error::InvalidResponse)
    }

    fn check_file_url(&self, url: &url::Url) -> Result<()> {
        #[cfg(test)]
        if url.scheme() == "http"
            && url.host_str() == Some("127.0.0.1")
            && self.default_base.starts_with("http://127.0.0.1:")
            && url.username().is_empty()
            && url.password().is_none()
        {
            return Ok(());
        }
        if url.scheme() != "https"
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(Error::InvalidResponse);
        }
        Ok(())
    }

    fn account_request(
        &self,
        requests: &mut Requests,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
    ) -> Result<HttpReply> {
        self.account_request_cancellable(requests, (method, path, body), &|| false)
    }

    fn account_request_cancellable(
        &self,
        requests: &mut Requests,
        request: (&str, &str, Option<&[u8]>),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<HttpReply> {
        check_cancelled(cancelled)?;
        let mut account = lock(&self.status).saved.clone();
        let mut refreshed = false;
        if let Some(saved) = &account {
            if saved.rejected {
                return Err(Error::CredentialsRejected);
            }
            if saved.token_expired() {
                account = Some(self.refresh_login(requests, saved, cancelled)?);
                refreshed = true;
            }
        }
        let reply = self.send_endpoint(requests, request, account.as_ref(), cancelled)?;
        if reply.status != 401 || account.is_none() {
            return Ok(reply);
        }
        if !refreshed && let Some(saved) = &account {
            let renewed = self.refresh_login(requests, saved, cancelled)?;
            let retry = self.send_endpoint(requests, request, Some(&renewed), cancelled)?;
            if retry.status != 401 {
                return Ok(retry);
            }
            self.reject_account(renewed)?;
        } else if let Some(account) = account {
            self.reject_account(account)?;
        }
        Err(Error::CredentialsRejected)
    }

    fn send_endpoint(
        &self,
        requests: &mut Requests,
        request: (&str, &str, Option<&[u8]>),
        account: Option<&Account>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<HttpReply> {
        let response = self.api_request(requests, request, account, cancelled);
        let transient = match &response {
            Ok(reply) => matches!(reply.status, 408 | 429 | 500 | 502 | 503 | 504),
            Err(error) => *error == Error::Unavailable,
        };
        if request.0 == "GET"
            && transient
            && requests.next_api.is_none_or(|until| {
                until.saturating_duration_since(Instant::now()) <= MAX_RETRY_WAIT
            })
        {
            return self.api_request(requests, request, account, cancelled);
        }
        response
    }

    fn login(
        &self,
        requests: &mut Requests,
        credentials: Credentials,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(Account, Quota)> {
        let body = serde_json::to_vec(&credentials).map_err(|_| Error::InvalidResponse)?;
        let reply = self.api_request(requests, ("POST", "/login", Some(&body)), None, cancelled)?;
        let login = json::<LoginReply>(&reply, |status| match status {
            400 | 401 | 403 | 422 => Error::CredentialsRejected,
            _ => Error::InvalidResponse,
        })?;
        api_base(&login.base_url)?;
        if login.token.is_empty() || !login.token.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(Error::InvalidResponse);
        }
        Ok((
            Account {
                credentials,
                token: login.token,
                api_host: login.base_url,
                issued_at: unix_now(),
                rejected: false,
            },
            Quota {
                allowed: login.user.allowed_downloads,
                ..Default::default()
            },
        ))
    }

    fn refresh_login(
        &self,
        requests: &mut Requests,
        stale: &Account,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Account> {
        match self.login(requests, stale.credentials.clone(), cancelled) {
            Ok((account, quota)) => {
                self.save_account(&account)?;
                let mut status = lock(&self.status);
                status.saved = Some(account.clone());
                status.quota.allowed = quota.allowed;
                Ok(account)
            }
            Err(Error::CredentialsRejected) => {
                self.reject_account(stale.clone())?;
                Err(Error::CredentialsRejected)
            }
            Err(error) => Err(error),
        }
    }

    fn reject_account(&self, mut account: Account) -> Result<()> {
        account.token.clear();
        account.rejected = true;
        *lock(&self.status) = AccountStatus {
            saved: Some(account.clone()),
            quota: Quota::default(),
        };
        self.save_account(&account)
    }

    fn save_account(&self, account: &Account) -> Result<()> {
        let data = serde_json::to_vec(account).map_err(|_| Error::Storage)?;
        write_private(&self.credentials_path, &data).map_err(|_| Error::Storage)
    }

    fn api_request(
        &self,
        requests: &mut Requests,
        (method, path, body): (&str, &str, Option<&[u8]>),
        account: Option<&Account>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<HttpReply> {
        if !self.available() {
            return Err(Error::Unconfigured);
        }
        wait_until(requests.next_api, cancelled)?;
        if path == "/login" {
            wait_until(requests.next_login, cancelled)?;
            requests.next_login = Some(Instant::now() + Duration::from_secs(1));
        }
        let (base, client) = match account {
            Some(account) if api_base(&account.api_host)? == VIP_API => {
                (self.vip_base.as_str(), &self.vip_api)
            }
            _ => (self.default_base.as_str(), &self.api),
        };
        let mut headers = vec![("Accept", "application/json")];
        if body.is_some() {
            headers.push(("Content-Type", "application/json"));
        }
        let auth;
        if let Some(account) = account {
            auth = format!("Bearer {}", account.token);
            // The scoped API client refuses cross-origin redirects.
            headers.push(("Authorization", &auth));
        }
        requests.next_api = Some(Instant::now() + REQUEST_SPACING);
        let reply = client
            .request_no_status_retry(
                method,
                &format!("{base}{path}"),
                &headers,
                body,
                SIZE_LIMIT + 1,
            )
            .map_err(|error| match error {
                crate::Error::InvalidLocation { .. } => Error::InvalidResponse,
                _ => Error::Unavailable,
            })?;
        update_window(&mut requests.next_api, &reply);
        if reply.body.len() as u64 > SIZE_LIMIT {
            return Err(Error::InvalidResponse);
        }
        Ok(reply)
    }
}

fn api_base(host: &str) -> Result<&'static str> {
    match host {
        "api.opensubtitles.com" => Ok(API),
        "vip-api.opensubtitles.com" => Ok(VIP_API),
        _ => Err(Error::InvalidResponse),
    }
}

#[derive(Deserialize)]
struct Data<T> {
    data: T,
}

#[derive(Deserialize)]
struct LoginReply {
    user: UserInfo,
    base_url: String,
    token: String,
}

#[derive(Deserialize)]
struct UserInfo {
    allowed_downloads: Option<u32>,
    remaining_downloads: Option<u32>,
    reset_time_utc: Option<String>,
}

#[derive(Deserialize)]
struct LinkReply {
    link: Option<String>,
    remaining: Option<u32>,
    reset_time_utc: Option<String>,
}

fn load_account(path: &std::path::Path) -> Result<Option<Account>> {
    let raw = match std::fs::read(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(Error::Storage),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|_| Error::Storage)?;
    }
    let account: Account = serde_json::from_slice(&raw).map_err(|_| Error::Storage)?;
    api_base(&account.api_host)?;
    Ok(Some(account))
}

fn json<T: DeserializeOwned>(reply: &HttpReply, otherwise: impl FnOnce(u16) -> Error) -> Result<T> {
    response_ok(reply, otherwise)?;
    serde_json::from_slice(&reply.body).map_err(|_| Error::InvalidResponse)
}

fn response_ok(reply: &HttpReply, otherwise: impl FnOnce(u16) -> Error) -> Result<()> {
    match reply.status {
        200..=299 => Ok(()),
        429 => Err(Error::RateLimited {
            retry_after: retry_after(reply).unwrap_or(Duration::from_secs(1)),
        }),
        408 | 500..=599 => Err(Error::Unavailable),
        status => Err(otherwise(status)),
    }
}

fn retry_after(reply: &HttpReply) -> Option<Duration> {
    let parse = |value: &str| {
        value
            .trim()
            .parse::<u64>()
            .ok()
            .or_else(|| Some(parse_http_date(value)?.saturating_sub(unix_now()).max(0) as u64))
            .map(Duration::from_secs)
    };
    reply
        .header("retry-after")
        .and_then(parse)
        .or_else(|| reply.header("ratelimit-reset").and_then(parse))
}

fn update_window(next: &mut Option<Instant>, reply: &HttpReply) -> Duration {
    let exhausted = ["ratelimit-remaining", "x-ratelimit-remaining-second"]
        .iter()
        .any(|name| reply.header(name).is_some_and(|s| s.trim() == "0"));
    let delay = if reply.status == 429 || exhausted || reply.header("retry-after").is_some() {
        retry_after(reply).unwrap_or(Duration::from_secs(1))
    } else {
        REQUEST_SPACING
    };
    *next = Instant::now().checked_add(delay);
    delay
}

fn wait_until(until: Option<Instant>, cancelled: &dyn Fn() -> bool) -> Result<()> {
    if let Some(until) = until {
        let remaining = || until.saturating_duration_since(Instant::now());
        if remaining() > MAX_RETRY_WAIT {
            return Err(Error::RateLimited {
                retry_after: remaining(),
            });
        }
        while !remaining().is_zero() {
            check_cancelled(cancelled)?;
            std::thread::sleep(remaining().min(Duration::from_millis(50)));
        }
    }
    check_cancelled(cancelled)
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<()> {
    if cancelled() {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

fn unix_now() -> i64 {
    system_time_to_unix(SystemTime::now())
}

#[cfg(test)]
mod tests;
