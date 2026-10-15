//! `scitok-learn`: harvest arXiv for one category and learn scitok domain data.

mod cache;
mod fulltext;
mod http;
mod learn;
mod oai;
mod refine;

use std::path::PathBuf;
use std::process::ExitCode;

use chrono::{Months, NaiveDate, Utc};
use clap::{Args, Parser, Subcommand};

use cache::Cache;
use http::{ARXIV_MIN_INTERVAL, Client, Rate, Ureq};
use oai::Filter;
use scitok::mine::MineConfig;

#[derive(Parser)]
#[command(version, about = "Harvest arXiv and learn scitok domain data (lexicon + statistics)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Harvest abstracts (and optionally full texts) into the cache.
    Fetch {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        fetch: FetchArgs,
    },
    /// Mine the cache into domain files (offline).
    Build {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        build: BuildArgs,
    },
    /// Review the mined lexicon and add variants with an LLM (writes reviewed.toml).
    Refine {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        refine: RefineArgs,
    },
    /// `fetch` followed by `build`.
    Learn {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        fetch: FetchArgs,
        #[command(flatten)]
        build: BuildArgs,
    },
}

#[derive(Args)]
struct Common {
    /// arXiv category, e.g. quant-ph, cond-mat.str-el, cs.ET.
    #[arg(long)]
    category: String,
    /// First day (YYYY-MM-DD) [default: six months before --until].
    #[arg(long)]
    from: Option<NaiveDate>,
    /// Last day (YYYY-MM-DD) [default: today].
    #[arg(long)]
    until: Option<NaiveDate>,
    #[arg(long, default_value = ".scitok-cache")]
    cache_dir: PathBuf,
    /// Also include papers whose primary category is another one.
    #[arg(long)]
    include_cross_lists: bool,
    /// Also include older papers that were revised in the date range.
    #[arg(long)]
    include_revisions: bool,
}

#[derive(Args)]
struct FetchArgs {
    /// Also download LaTeX sources (one request per paper; slow at arXiv's rate).
    #[arg(long)]
    full_text: bool,
    /// Request rate as N/<duration>; arXiv asks for at most 1/3s.
    #[arg(long, default_value = "1/3s")]
    rate: Rate,
    /// Stop once this many new papers are cached (checked per OAI page of
    /// ~1000 records) and after this many full-text downloads; rerun to continue.
    #[arg(long)]
    max_papers: Option<usize>,
    /// Contact email for the User-Agent, as arXiv requests [env: SCITOK_CONTACT].
    #[arg(long, env = "SCITOK_CONTACT")]
    contact: Option<String>,
    #[arg(long, default_value = oai::DEFAULT_OAI_URL)]
    oai_url: String,
    #[arg(long, default_value = fulltext::DEFAULT_SOURCE_URL)]
    source_url: String,
}

#[derive(Args)]
struct BuildArgs {
    /// Domain directory to write [default: crates/scitok/data/domains/<category>].
    #[arg(long)]
    out: Option<PathBuf>,
    /// Minimum number of documents for acronyms and collocations.
    #[arg(long, default_value_t = MineConfig::default().min_count)]
    min_count: u32,
    /// Minimum number of documents for eponyms (`X-Y theorem`).
    #[arg(long, default_value_t = MineConfig::default().min_eponym_count)]
    min_eponym_count: u32,
    /// Minimum normalized PMI for collocations.
    #[arg(long, default_value_t = MineConfig::default().min_npmi)]
    min_npmi: f32,
    /// Maximum number of collocations kept.
    #[arg(long, default_value_t = MineConfig::default().max_phrases)]
    max_phrases: usize,
    /// Longest collocation, in words.
    #[arg(long, default_value_t = MineConfig::default().max_ngram)]
    max_ngram: usize,
}

#[derive(Args)]
struct RefineArgs {
    /// Domain directory with curated.toml and mined.toml [default: crates/scitok/data/domains/<category>].
    #[arg(long)]
    out: Option<PathBuf>,
    /// Chat model; any model supporting JSON-schema structured output.
    #[arg(long, default_value = "gpt-6.1-sol")]
    model: String,
    /// OpenAI-compatible API base URL. The key is read from OPENAI_API_KEY.
    #[arg(long, default_value = "https://api.openai.com/v1")]
    base_url: String,
    /// Mined entries per review request.
    #[arg(long, default_value_t = refine::RefineConfig::default().review_batch)]
    review_batch: usize,
    /// Concepts per expansion request.
    #[arg(long, default_value_t = refine::RefineConfig::default().expand_batch)]
    expand_batch: usize,
    /// Minimum number of documents a proposed variant must occur in.
    #[arg(long, default_value_t = refine::RefineConfig::default().min_attested)]
    min_attested: u32,
    /// API request rate.
    #[arg(long, default_value = "1/1s")]
    rate: Rate,
    /// Count the requests that would be made (cached ones are free) without calling the API.
    #[arg(long)]
    dry_run: bool,
}

impl Common {
    fn filter(&self) -> Result<Filter, String> {
        let until = self.until.unwrap_or_else(|| Utc::now().date_naive());
        let from = self.from.unwrap_or_else(|| until.checked_sub_months(Months::new(6)).unwrap());
        if from > until {
            return Err(format!("--from {from} is after --until {until}"));
        }
        oai::set_spec(&self.category)?;
        Ok(Filter {
            category: self.category.clone(),
            from,
            until,
            include_cross_lists: self.include_cross_lists,
            include_revisions: self.include_revisions,
        })
    }
}

fn fetch(common: &Common, args: &FetchArgs) -> Result<(), String> {
    let filter = common.filter()?;
    if args.rate.interval() < ARXIV_MIN_INTERVAL {
        eprintln!(
            "warning: --rate {} is faster than arXiv's policy of one request every {}s; you may be blocked",
            args.rate,
            ARXIV_MIN_INTERVAL.as_secs()
        );
    }
    if args.contact.is_none() {
        eprintln!("note: set --contact or SCITOK_CONTACT to include a contact email in the User-Agent");
    }
    eprintln!("harvesting {} from {} to {}", filter.category, filter.from, filter.until);
    let mut cache = Cache::open(&common.cache_dir, &filter.category)?;
    let mut client = Client::new(Ureq::new(args.contact.as_deref()), args.rate);
    let stats = oai::harvest(&mut client, &mut cache, &args.oai_url, &filter, args.max_papers)?;
    eprintln!(
        "abstracts: {} pages, {} records seen, {} accepted, {} new; {} cached{}",
        stats.pages,
        stats.seen,
        stats.accepted,
        stats.added,
        cache.len(),
        if stats.complete { "" } else { " (incomplete, rerun to continue)" }
    );
    if args.full_text {
        eprintln!(
            "note: for bulk full-text corpora prefer arXiv's requester-pays S3 bucket \
             (https://info.arxiv.org/help/bulk_data_s3.html)"
        );
        let records: Vec<_> = cache.records()?.into_iter().filter(|r| filter.accepts(r)).collect();
        let stats = fulltext::fetch_all(&mut client, &cache, &args.source_url, &records, args.max_papers)?;
        eprintln!("full text: {} fetched, {} without LaTeX, {} already cached", stats.fetched, stats.skipped, stats.cached);
    }
    Ok(())
}

fn build(common: &Common, args: &BuildArgs) -> Result<(), String> {
    let filter = common.filter()?;
    let cache = Cache::open(&common.cache_dir, &filter.category)?;
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from("crates/scitok/data/domains").join(&filter.category));
    let cfg = MineConfig {
        min_count: args.min_count,
        min_eponym_count: args.min_eponym_count,
        min_npmi: args.min_npmi,
        max_phrases: args.max_phrases,
        max_ngram: args.max_ngram.clamp(2, 6),
        ..MineConfig::default()
    };
    let report = learn::build(&cache, &filter, &out, cfg)?;
    eprintln!(
        "wrote {}: {} mined terms from {} abstracts and {} full texts (avg length {})",
        out.display(),
        report.terms,
        report.abstracts,
        report.fulltexts,
        report.avg_len
    );
    Ok(())
}

fn refine(common: &Common, args: &RefineArgs) -> Result<(), String> {
    let filter = common.filter()?;
    let cache = Cache::open(&common.cache_dir, &filter.category)?;
    let out = args.out.clone().unwrap_or_else(|| PathBuf::from("crates/scitok/data/domains").join(&filter.category));
    let mut client = Client::new(Ureq::new(None), args.rate);
    // completions are slow and billed: fewer retries than for arXiv
    client.max_attempts = 3;
    let mut llm = refine::Llm {
        client,
        base_url: args.base_url.clone(),
        model: args.model.clone(),
        api_key: std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()),
        cache_dir: common.cache_dir.join(&filter.category).join("llm"),
        dry_run: args.dry_run,
        calls: 0,
        cached: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
    };
    let cfg = refine::RefineConfig { review_batch: args.review_batch, expand_batch: args.expand_batch, min_attested: args.min_attested };
    let r = refine::refine(&mut llm, &cache, &filter, &out, &cfg)?;
    if args.dry_run {
        eprintln!("dry run: {} API requests needed ({} cached)", llm.calls, llm.cached);
        return Ok(());
    }
    eprintln!(
        "review: {} kept ({} renamed), {} dropped, {} merged, {} undecided\n\
         expand: {} proposed, {} accepted, {} not in corpus, {} conflicting, {} already known, {} too broad\n\
         API: {} requests ({} cached), {} prompt + {} completion tokens\n\
         avg_len {} written to stats.toml; wrote {}",
        r.kept, r.renamed, r.dropped, r.merged, r.undecided,
        r.proposed, r.accepted, r.unattested, r.conflicting, r.duplicates, r.too_broad,
        llm.calls, llm.cached, llm.prompt_tokens, llm.completion_tokens,
        r.avg_len, out.join("reviewed.toml").display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Fetch { common, fetch: f } => fetch(&common, &f),
        Command::Build { common, build: b } => build(&common, &b),
        Command::Refine { common, refine: r } => refine(&common, &r),
        Command::Learn { common, fetch: f, build: b } => fetch(&common, &f).and_then(|()| build(&common, &b)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
