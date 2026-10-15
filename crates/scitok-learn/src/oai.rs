//! arXiv OAI-PMH harvesting (`ListRecords`, `metadataPrefix=arXiv`).

use chrono::NaiveDate;
use quick_xml::Reader;
use quick_xml::events::Event;

use crate::cache::{Cache, HarvestState, Record};
use crate::http::{Client, Clock, Transport};

pub const DEFAULT_OAI_URL: &str = "https://oaipmh.arxiv.org/oai";

const PHYSICS_ARCHIVES: &[&str] = &[
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph", "nlin",
    "nucl-ex", "nucl-th", "physics", "quant-ph",
];

/// OAI set for an arXiv category: `quant-ph` → `physics:quant-ph`,
/// `cond-mat.str-el` → `physics:cond-mat:str-el`, `cs.ET` → `cs:cs:ET`.
pub fn set_spec(category: &str) -> Result<String, String> {
    let (archive, sub) = match category.split_once('.') {
        Some((a, s)) => (a, Some(s)),
        None => (category, None),
    };
    let valid = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-');
    if !valid(archive) || sub.is_some_and(|s| !valid(s)) {
        return Err(format!("invalid arXiv category {category:?}"));
    }
    let group = if PHYSICS_ARCHIVES.contains(&archive) { "physics" } else { archive };
    Ok(match sub {
        Some(sub) => format!("{group}:{archive}:{sub}"),
        None if group == archive => archive.to_owned(),
        None => format!("{group}:{archive}"),
    })
}

#[derive(Debug, Default)]
pub struct Page {
    pub records: Vec<Record>,
    pub token: Option<String>,
    pub complete_list_size: Option<u64>,
}

/// Parse a `ListRecords` response. `noRecordsMatch` is an empty page; other
/// OAI errors are errors. Deleted records are skipped.
pub fn parse_page(xml: &str) -> Result<Page, String> {
    let mut reader = Reader::from_str(xml);
    let mut page = Page::default();
    let mut path: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut record = Record::default();
    let mut deleted = false;
    let mut error_code: Option<String> = None;
    loop {
        let event = reader.read_event().map_err(|e| format!("OAI-PMH XML error at {}: {e}", reader.buffer_position()))?;
        match event {
            Event::Start(e) => {
                let name = e.local_name().as_ref().to_owned();
                match name.as_str() {
                    "record" => {
                        record = Record::default();
                        deleted = false;
                    }
                    "header" => deleted = attr(&e, "status").as_deref() == Some("deleted"),
                    "resumptionToken" => {
                        page.complete_list_size = attr(&e, "completeListSize").and_then(|s| s.parse().ok())
                    }
                    "error" => error_code = Some(attr(&e, "code").unwrap_or_default()),
                    _ => {}
                }
                path.push(name);
                text.clear();
            }
            Event::Empty(e) => {
                if e.local_name().as_ref() == "error" {
                    error_code = Some(attr(&e, "code").unwrap_or_default());
                }
            }
            Event::Text(e) => text.push_str(&e.html_content()),
            Event::CData(e) => text.push_str(&e),
            Event::GeneralRef(r) => {
                if let Ok(Some(c)) = r.resolve_char_ref() {
                    text.push(c);
                } else {
                    text.push_str(match r.html_content().as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        _ => "",
                    });
                }
            }
            Event::End(_) => {
                let name = path.pop().unwrap_or_default();
                let parent = path.last().map(String::as_str).unwrap_or("");
                let value = || collapse_ws(&text);
                match (parent, name.as_str()) {
                    ("header", "datestamp") => record.datestamp = value(),
                    ("arXiv", "id") => record.id = value(),
                    ("arXiv", "created") => record.created = Some(value()),
                    ("arXiv", "categories") => record.categories = text.split_whitespace().map(str::to_owned).collect(),
                    ("arXiv", "title") => record.title = value(),
                    ("arXiv", "abstract") => record.abstract_ = value(),
                    (_, "resumptionToken") => page.token = Some(text.trim().to_owned()).filter(|t| !t.is_empty()),
                    (_, "record") if !deleted && !record.id.is_empty() => page.records.push(std::mem::take(&mut record)),
                    _ => {}
                }
                text.clear();
            }
            Event::Eof => break,
            _ => {}
        }
    }
    match error_code.as_deref() {
        None | Some("noRecordsMatch") => Ok(page),
        Some(code) => Err(format!("OAI-PMH error {code:?}")),
    }
}

fn attr(e: &quick_xml::events::BytesStart, name: &str) -> Option<String> {
    e.try_get_attribute(name).ok().flatten().map(|a| a.value.into_owned())
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Debug)]
pub struct Filter {
    pub category: String,
    pub from: NaiveDate,
    pub until: NaiveDate,
    pub include_cross_lists: bool,
    pub include_revisions: bool,
}

impl Filter {
    /// OAI datestamps select records new *or updated* in the range. Unless
    /// revisions are wanted, keep only papers whose arXiv id (YYMM) is not
    /// older than the start of the range, i.e. first submitted in it.
    pub fn accepts(&self, r: &Record) -> bool {
        let in_range = NaiveDate::parse_from_str(&r.datestamp, "%Y-%m-%d").is_ok_and(|d| d >= self.from && d <= self.until);
        let category = if self.include_cross_lists {
            r.categories.iter().any(|c| c == &self.category)
        } else {
            r.categories.first() == Some(&self.category)
        };
        in_range && category && (self.include_revisions || self.is_new(&r.id))
    }

    fn is_new(&self, id: &str) -> bool {
        // new-style ids: YYMM.NNNNN; old-style (quant-ph/0101001) predate 2007
        let Some(yymm) = id.split('.').next().filter(|p| p.len() == 4 && p.bytes().all(|b| b.is_ascii_digit())) else {
            return false;
        };
        let start = self.from.format("%y%m").to_string();
        yymm >= start.as_str()
    }
}

#[derive(Debug, Default)]
pub struct HarvestStats {
    pub pages: usize,
    pub seen: usize,
    pub accepted: usize,
    pub added: usize,
    pub complete: bool,
}

/// Harvest the OAI set of `filter.category` into the cache (all records;
/// `filter` only feeds the statistics), resuming an interrupted harvest of
/// the same range. Stops after `max_new` newly cached records, saving the
/// resumption token so a later run continues.
pub fn harvest<T: Transport, C: Clock>(
    client: &mut Client<T, C>,
    cache: &mut Cache,
    base_url: &str,
    filter: &Filter,
    max_new: Option<usize>,
) -> Result<HarvestStats, String> {
    let set = set_spec(&filter.category)?;
    let mut state = HarvestState {
        set: set.clone(),
        from: filter.from.to_string(),
        until: filter.until.to_string(),
        token: None,
        done: false,
    };
    if let Some(prev) = cache.state()
        && prev.set == state.set
        && prev.from == state.from
        && prev.until == state.until
    {
        if prev.done {
            eprintln!("harvest of {set} {}..{} already complete", state.from, state.until);
            return Ok(HarvestStats { complete: true, ..HarvestStats::default() });
        }
        if prev.token.is_some() {
            eprintln!("resuming interrupted harvest of {set}");
            state.token = prev.token;
        }
    }
    let mut stats = HarvestStats::default();
    loop {
        let url = match &state.token {
            Some(token) => format!("{base_url}?verb=ListRecords&resumptionToken={}", url_encode(token)),
            None => format!(
                "{base_url}?verb=ListRecords&metadataPrefix=arXiv&set={}&from={}&until={}",
                url_encode(&set),
                state.from,
                state.until
            ),
        };
        let resp = client.get(&url)?;
        if resp.status != 200 {
            return Err(format!("GET {url}: HTTP {}", resp.status));
        }
        let page = match parse_page(&String::from_utf8_lossy(&resp.body)) {
            // expired token from an earlier run: start over (the cache dedupes)
            Err(e) if state.token.is_some() && e.contains("badResumptionToken") => {
                eprintln!("saved resumption token was rejected; restarting the harvest");
                state.token = None;
                continue;
            }
            page => page?,
        };
        stats.pages += 1;
        stats.seen += page.records.len();
        stats.accepted += page.records.iter().filter(|r| filter.accepts(r)).count();
        // Cache everything in the set; `build` applies the filter, so later
        // runs with --include-cross-lists/--include-revisions need no refetch.
        stats.added += cache.append(&page.records)?;
        state.token = page.token;
        state.done = state.token.is_none();
        cache.save_state(&state)?;
        eprintln!(
            "page {}: {} records seen, {} accepted, {} cached in total{}",
            stats.pages,
            stats.seen,
            stats.accepted,
            cache.len(),
            page.complete_list_size.map(|n| format!(" (list size {n})")).unwrap_or_default()
        );
        if state.done {
            stats.complete = true;
            return Ok(stats);
        }
        if max_new.is_some_and(|max| stats.added >= max) {
            eprintln!("reached --max-papers; run again to continue from the saved resumption token");
            return Ok(stats);
        }
    }
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::Rate;
    use crate::http::testing::{FakeClock, FakeTransport};

    const PAGE1: &str = include_str!("../tests/fixtures/oai_page1.xml");
    const PAGE2: &str = include_str!("../tests/fixtures/oai_page2.xml");
    const NO_RECORDS: &str = include_str!("../tests/fixtures/oai_no_records.xml");

    fn filter() -> Filter {
        Filter {
            category: "quant-ph".into(),
            from: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            until: NaiveDate::from_ymd_opt(2026, 10, 15).unwrap(),
            include_cross_lists: false,
            include_revisions: false,
        }
    }

    #[test]
    fn set_specs() {
        assert_eq!(set_spec("quant-ph").unwrap(), "physics:quant-ph");
        assert_eq!(set_spec("cond-mat.str-el").unwrap(), "physics:cond-mat:str-el");
        assert_eq!(set_spec("cs.ET").unwrap(), "cs:cs:ET");
        assert_eq!(set_spec("cs").unwrap(), "cs");
        assert_eq!(set_spec("physics.optics").unwrap(), "physics:physics:optics");
        assert!(set_spec("quant ph").is_err());
    }

    #[test]
    fn parses_records() {
        let page = parse_page(PAGE1).unwrap();
        assert_eq!(page.token.as_deref(), Some("6960524|1001"));
        assert_eq!(page.complete_list_size, Some(1500));
        assert_eq!(page.records.len(), 2);
        let r = &page.records[0];
        assert_eq!(r.id, "2610.01234");
        assert_eq!(r.datestamp, "2026-10-08");
        assert_eq!(r.categories, vec!["quant-ph", "cond-mat.str-el"]);
        assert_eq!(r.title, "Lieb-Robinson bounds & scrambling in long-range spin chains");
        assert!(r.abstract_.starts_with("We derive a Lieb-Robinson bound"), "{}", r.abstract_);
        assert!(r.abstract_.contains("$\\alpha > 2d$"));
    }

    #[test]
    fn deleted_records_and_errors() {
        let page = parse_page(PAGE2).unwrap();
        assert_eq!(page.records.len(), 2);
        assert_eq!(page.token, None);
        assert!(parse_page(NO_RECORDS).unwrap().records.is_empty());
        let bad = "<OAI-PMH><error code=\"badArgument\">bad</error></OAI-PMH>";
        assert!(parse_page(bad).is_err());
    }

    #[test]
    fn filters_primary_category_and_revisions() {
        let p1 = parse_page(PAGE1).unwrap();
        let p2 = parse_page(PAGE2).unwrap();
        let all: Vec<&Record> = p1.records.iter().chain(&p2.records).collect();
        let ids = |f: &Filter| all.iter().filter(|r| f.accepts(r)).map(|r| r.id.as_str()).collect::<Vec<_>>();
        assert_eq!(ids(&filter()), vec!["2610.01234", "2609.04321"]);
        let f = Filter { include_cross_lists: true, ..filter() };
        assert_eq!(ids(&f), vec!["2610.01234", "2610.05555", "2609.04321"]);
        let f = Filter { include_revisions: true, ..filter() };
        assert_eq!(ids(&f), vec!["2610.01234", "2212.06976", "2609.04321"]);
        let f = Filter { until: NaiveDate::from_ymd_opt(2026, 10, 1).unwrap(), ..filter() };
        assert_eq!(ids(&f), vec!["2609.04321"]);
    }

    fn client(t: FakeTransport, clock: FakeClock) -> Client<FakeTransport, FakeClock> {
        let mut c = Client::with_clock(t, Rate::default(), clock);
        c.verbose = false;
        c
    }

    #[test]
    fn harvests_pages_and_resumes() {
        let root = std::env::temp_dir().join(format!("scitok-learn-harvest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let clock = FakeClock::new();

        // first run: stop after the first page
        let mut t = FakeTransport::new(clock.clone());
        t.route("set=physics%3Aquant-ph&from=2026-04-01&until=2026-10-15", 200, PAGE1.as_bytes());
        let mut cache = Cache::open(&root, "quant-ph").unwrap();
        let stats = harvest(&mut client(t, clock.clone()), &mut cache, "http://oai", &filter(), Some(1)).unwrap();
        assert_eq!((stats.pages, stats.accepted, stats.added, stats.complete), (1, 1, 2, false));
        assert_eq!(cache.state().unwrap().token.as_deref(), Some("6960524|1001"));

        // second run resumes from the token
        let mut t = FakeTransport::new(clock.clone());
        t.route("resumptionToken=6960524%7C1001", 200, PAGE2.as_bytes());
        let log = t.log.clone();
        let mut cache = Cache::open(&root, "quant-ph").unwrap();
        let stats = harvest(&mut client(t, clock.clone()), &mut cache, "http://oai", &filter(), None).unwrap();
        assert_eq!((stats.pages, stats.accepted, stats.added, stats.complete), (1, 1, 2, true));
        assert_eq!(log.borrow().len(), 1);
        assert_eq!(cache.len(), 4);

        // third run: nothing to do
        let t = FakeTransport::new(clock.clone());
        let stats = harvest(&mut client(t, clock), &mut cache, "http://oai", &filter(), None).unwrap();
        assert!(stats.complete && stats.pages == 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn restarts_on_rejected_token() {
        let root = std::env::temp_dir().join(format!("scitok-learn-badtoken-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut cache = Cache::open(&root, "quant-ph").unwrap();
        let f = filter();
        let stale = HarvestState {
            set: "physics:quant-ph".into(),
            from: f.from.to_string(),
            until: f.until.to_string(),
            token: Some("stale".into()),
            done: false,
        };
        cache.save_state(&stale).unwrap();
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        let bad = "<OAI-PMH><error code=\"badResumptionToken\">expired</error></OAI-PMH>";
        t.route("resumptionToken=stale", 200, bad.as_bytes());
        t.route("set=physics%3Aquant-ph", 200, PAGE2.as_bytes());
        let stats = harvest(&mut client(t, clock), &mut cache, "http://oai", &f, None).unwrap();
        assert!(stats.complete);
        assert_eq!(cache.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
