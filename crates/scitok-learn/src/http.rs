//! Rate-limited, retrying HTTP client over an injectable transport.

use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

/// arXiv asks automated clients for at most one request every three seconds.
pub const ARXIV_MIN_INTERVAL: Duration = Duration::from_secs(3);

pub struct Response {
    pub status: u16,
    pub retry_after: Option<Duration>,
    pub content_type: Option<String>,
    pub body: Vec<u8>,
}

pub trait Transport {
    fn get(&mut self, url: &str) -> Result<Response, String>;
    /// POST `body` with extra `headers` (e.g. authorization; never logged).
    fn post(&mut self, url: &str, headers: &[(&str, &str)], body: &[u8]) -> Result<Response, String>;
}

/// Time source, so tests can run on virtual time.
pub trait Clock {
    fn now(&self) -> Instant;
    fn sleep(&mut self, d: Duration);
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep(&mut self, d: Duration) {
        std::thread::sleep(d)
    }
}

/// `N/<duration>`, e.g. `1/3s`, `2/1s`, `1/500ms`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rate {
    pub requests: u32,
    pub per: Duration,
}

impl Rate {
    pub fn interval(&self) -> Duration {
        self.per / self.requests
    }
}

impl Default for Rate {
    fn default() -> Self {
        Rate { requests: 1, per: ARXIV_MIN_INTERVAL }
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}ms", self.requests, self.per.as_millis())
    }
}

impl FromStr for Rate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let err = || format!("invalid rate {s:?}, expected e.g. 1/3s, 2/1s or 1/500ms");
        let (n, per) = s.split_once('/').ok_or_else(err)?;
        let requests: u32 = n.trim().parse().map_err(|_| err())?;
        let per = per.trim();
        let (num, unit) = per.split_at(per.find(|c: char| c.is_ascii_alphabetic()).ok_or_else(err)?);
        let num: f64 = if num.is_empty() { 1.0 } else { num.parse().map_err(|_| err())? };
        let secs = match unit {
            "ms" => num / 1000.0,
            "s" => num,
            "m" | "min" => num * 60.0,
            _ => return Err(err()),
        };
        if requests == 0 || secs <= 0.0 {
            return Err(err());
        }
        Ok(Rate { requests, per: Duration::try_from_secs_f64(secs).map_err(|_| err())? })
    }
}

pub struct Client<T: Transport, C: Clock = SystemClock> {
    transport: T,
    clock: C,
    interval: Duration,
    last: Option<Instant>,
    pub max_attempts: u32,
    pub verbose: bool,
}

impl<T: Transport> Client<T> {
    pub fn new(transport: T, rate: Rate) -> Self {
        Client::with_clock(transport, rate, SystemClock)
    }
}

impl<T: Transport, C: Clock> Client<T, C> {
    pub fn with_clock(transport: T, rate: Rate, clock: C) -> Self {
        Client { transport, clock, interval: rate.interval(), last: None, max_attempts: 6, verbose: true }
    }

    /// GET with rate limiting. Retries on 429/503 (honoring `Retry-After`),
    /// other 5xx and transport errors, with exponential backoff. Other
    /// statuses (including 404) are returned to the caller.
    pub fn get(&mut self, url: &str) -> Result<Response, String> {
        self.send("GET", url, |t| t.get(url))
    }

    /// POST with the same rate limiting and retries as [`get`](Self::get).
    pub fn post(&mut self, url: &str, headers: &[(&str, &str)], body: &[u8]) -> Result<Response, String> {
        self.send("POST", url, |t| t.post(url, headers, body))
    }

    fn send(
        &mut self,
        method: &str,
        url: &str,
        mut request: impl FnMut(&mut T) -> Result<Response, String>,
    ) -> Result<Response, String> {
        let mut backoff = self.interval.max(Duration::from_secs(1));
        let mut last_failure = String::new();
        for attempt in 1..=self.max_attempts {
            self.wait_turn();
            let result = request(&mut self.transport);
            self.last = Some(self.clock.now());
            let wait = match result {
                Ok(resp) => {
                    self.log(&format!("{method} {url} -> {}", resp.status));
                    let body = String::from_utf8_lossy(&resp.body);
                    // an exhausted quota is reported as 429 but will not recover
                    if resp.status == 429 && body.contains("insufficient_quota") {
                        return Ok(resp);
                    }
                    if resp.status == 429 || resp.status >= 500 {
                        last_failure = format!("HTTP {}: {}", resp.status, body.chars().take(300).collect::<String>());
                    }
                    if resp.status == 429 || resp.status == 503 {
                        resp.retry_after.unwrap_or(backoff)
                    } else if resp.status >= 500 {
                        backoff
                    } else {
                        return Ok(resp);
                    }
                }
                Err(e) => {
                    self.log(&format!("{method} {url} failed: {e}"));
                    last_failure = e;
                    backoff
                }
            };
            if attempt < self.max_attempts {
                self.log(&format!("retrying in {:.1}s (attempt {}/{})", wait.as_secs_f64(), attempt + 1, self.max_attempts));
                self.clock.sleep(wait);
                backoff *= 2;
            }
        }
        Err(format!("{method} {url}: giving up after {} attempts; last: {last_failure}", self.max_attempts))
    }

    fn wait_turn(&mut self) {
        if let Some(last) = self.last {
            let elapsed = self.clock.now().saturating_duration_since(last);
            if elapsed < self.interval {
                self.clock.sleep(self.interval - elapsed);
            }
        }
    }

    fn log(&self, msg: &str) {
        if self.verbose {
            eprintln!("[{}] {msg}", chrono::Utc::now().format("%H:%M:%S%.3f"));
        }
    }
}

pub struct Ureq {
    agent: ureq::Agent,
    user_agent: String,
}

impl Ureq {
    pub fn new(contact: Option<&str>) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            // LLM completions can take minutes
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into();
        let version = env!("CARGO_PKG_VERSION");
        let user_agent = match contact {
            Some(c) => format!("scitok-learn/{version} (mailto:{c})"),
            None => format!("scitok-learn/{version}"),
        };
        Ureq { agent, user_agent }
    }
}

/// Response bodies above this size are rejected (e-prints can be large).
const MAX_BODY: u64 = 64 * 1024 * 1024;

impl Transport for Ureq {
    fn get(&mut self, url: &str) -> Result<Response, String> {
        let resp = self.agent.get(url).header("User-Agent", &self.user_agent).call().map_err(|e| e.to_string())?;
        read_response(resp)
    }

    fn post(&mut self, url: &str, headers: &[(&str, &str)], body: &[u8]) -> Result<Response, String> {
        let mut req = self.agent.post(url).header("User-Agent", &self.user_agent);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        read_response(req.send(body).map_err(|e| e.to_string())?)
    }
}

fn read_response(mut resp: ureq::http::Response<ureq::Body>) -> Result<Response, String> {
    let header = |name: &str| resp.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let retry_after = header("retry-after").and_then(|v| v.trim().parse::<u64>().ok()).map(Duration::from_secs);
    let content_type = header("content-type");
    let status = resp.status().as_u16();
    let body = resp.body_mut().with_config().limit(MAX_BODY).read_to_vec().map_err(|e| e.to_string())?;
    Ok(Response { status, retry_after, content_type, body })
}

#[cfg(test)]
pub mod testing {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    /// Virtual clock: `sleep` advances time instantly.
    #[derive(Clone)]
    pub struct FakeClock {
        pub start: Instant,
        pub offset: Rc<RefCell<Duration>>,
    }

    impl FakeClock {
        pub fn new() -> Self {
            FakeClock { start: Instant::now(), offset: Rc::default() }
        }

        pub fn elapsed(&self) -> Duration {
            *self.offset.borrow()
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            self.start + *self.offset.borrow()
        }

        fn sleep(&mut self, d: Duration) {
            *self.offset.borrow_mut() += d;
        }
    }

    /// Serves canned responses by URL substring, recording request times.
    pub struct FakeTransport {
        pub routes: Vec<(String, VecDeque<Response>)>,
        pub log: Rc<RefCell<Vec<(Duration, String)>>>,
        /// Bodies of POST requests, in order.
        pub bodies: Rc<RefCell<Vec<Vec<u8>>>>,
        pub clock: FakeClock,
    }

    impl FakeTransport {
        pub fn new(clock: FakeClock) -> Self {
            FakeTransport { routes: Vec::new(), log: Rc::default(), bodies: Rc::default(), clock }
        }

        pub fn route(&mut self, pattern: &str, status: u16, body: &[u8]) -> &mut Self {
            let resp = Response { status, retry_after: None, content_type: None, body: body.to_vec() };
            self.push(pattern, resp)
        }

        pub fn push(&mut self, pattern: &str, resp: Response) -> &mut Self {
            match self.routes.iter_mut().find(|(p, _)| p == pattern) {
                Some((_, q)) => q.push_back(resp),
                None => self.routes.push((pattern.to_owned(), VecDeque::from([resp]))),
            }
            self
        }
    }

    impl Transport for FakeTransport {
        fn post(&mut self, url: &str, _headers: &[(&str, &str)], body: &[u8]) -> Result<Response, String> {
            self.bodies.borrow_mut().push(body.to_vec());
            self.get(url)
        }

        fn get(&mut self, url: &str) -> Result<Response, String> {
            self.log.borrow_mut().push((self.clock.elapsed(), url.to_owned()));
            let (_, queue) = self
                .routes
                .iter_mut()
                .find(|(p, _)| url.contains(p.as_str()))
                .ok_or_else(|| format!("no fake route for {url}"))?;
            // the last response for a route repeats
            if queue.len() > 1 {
                Ok(queue.pop_front().unwrap())
            } else {
                let r = &queue[0];
                Ok(Response { status: r.status, retry_after: r.retry_after, content_type: r.content_type.clone(), body: r.body.clone() })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn parses_rates() {
        assert_eq!("1/3s".parse::<Rate>().unwrap().interval(), Duration::from_secs(3));
        assert_eq!("2/1s".parse::<Rate>().unwrap().interval(), Duration::from_millis(500));
        assert_eq!("1/500ms".parse::<Rate>().unwrap().interval(), Duration::from_millis(500));
        assert_eq!("1/s".parse::<Rate>().unwrap().interval(), Duration::from_secs(1));
        assert!("0/1s".parse::<Rate>().is_err());
        assert!("fast".parse::<Rate>().is_err());
        assert!("1/100000000000000000000s".parse::<Rate>().is_err());
    }

    #[test]
    fn spaces_requests_and_honors_retry_after() {
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        t.route("/ok", 200, b"fine");
        t.push("/busy", Response { status: 503, retry_after: Some(Duration::from_secs(10)), content_type: None, body: vec![] });
        t.route("/busy", 200, b"done");
        let log = t.log.clone();
        let mut client = Client::with_clock(t, Rate::default(), clock.clone());
        client.verbose = false;
        client.get("http://x/ok").unwrap();
        client.get("http://x/ok").unwrap();
        let r = client.get("http://x/busy").unwrap();
        assert_eq!(r.body, b"done");
        let times: Vec<u64> = log.borrow().iter().map(|(t, _)| t.as_secs()).collect();
        // 3 s spacing, then a 10 s Retry-After wait
        assert_eq!(times, vec![0, 3, 6, 16]);
    }

    #[test]
    fn exhausted_quota_is_not_retried() {
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        t.route("/q", 429, b"{\"error\": {\"code\": \"insufficient_quota\"}}");
        let log = t.log.clone();
        let mut client = Client::with_clock(t, Rate::default(), clock);
        client.verbose = false;
        assert_eq!(client.post("http://x/q", &[], b"{}").unwrap().status, 429);
        assert_eq!(log.borrow().len(), 1);
    }

    #[test]
    fn gives_up_eventually() {
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        t.route("/down", 500, b"");
        let mut client = Client::with_clock(t, Rate::default(), clock);
        client.verbose = false;
        client.max_attempts = 3;
        let Err(err) = client.get("http://x/down") else { panic!("expected an error") };
        assert!(err.contains("last: HTTP 500"), "{err}");
    }
}
