//! LLM refinement of a mined domain lexicon (`scitok-learn refine`).
//!
//! 1. Review: every mined entry is shown to the model with its document
//!    frequency and corpus contexts, and kept, dropped (generic phrasing) or
//!    merged into a synonymous concept, optionally with a cleaner name.
//! 2. Discover: the model lists domain concepts in a sample of abstracts that
//!    the lexicon lacks (rare, long or low-NPMI phrases the miner cannot find).
//!    Only multi-word concepts whose variants occur in the corpus and belong
//!    to no other concept are added.
//! 3. Expand: curated and kept concepts get additional variants (synonyms,
//!    acronym/long form, spellings). Proposals are only accepted if they occur
//!    in the corpus and do not already belong to another concept, so
//!    hallucinated or conflicting variants never reach the lexicon.
//!
//! The result is written to `reviewed.toml`, which domains load instead of
//! `mined.toml`. Responses are cached on disk, so reruns are free.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use scitok::domain::{FORMAT_VERSION, LexiconFile};
use scitok::lexer::{Atom, Kind, lex};
use scitok::lexicon::{Lexicon, TermDef};
use scitok::normalize::normalize;
use scitok::{Config, Input, Tokenizer};

use crate::cache::Cache;
use crate::http::{Client, Clock, Transport};
use crate::learn::document;
use crate::oai::Filter;

pub struct RefineConfig {
    pub review_batch: usize,
    pub expand_batch: usize,
    /// Minimum number of documents a proposed variant must occur in.
    pub min_attested: u32,
    /// Abstracts sampled for discovering missing concepts (0 disables it).
    pub discover_docs: usize,
    /// Abstracts per discovery request.
    pub discover_batch: usize,
}

impl Default for RefineConfig {
    fn default() -> Self {
        RefineConfig { review_batch: 40, expand_batch: 30, min_attested: 2, discover_docs: 2000, discover_batch: 25 }
    }
}

/// Chat-completions client for an OpenAI-compatible API with an on-disk cache.
pub struct Llm<T: Transport, C: Clock> {
    pub client: Client<T, C>,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub cache_dir: PathBuf,
    pub dry_run: bool,
    pub calls: usize,
    pub cached: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl<T: Transport, C: Clock> Llm<T, C> {
    /// Structured completion: returns the JSON object matching `schema` and
    /// accepted by `valid`, or `None` in a dry run. Only valid replies are
    /// cached; invalid or incomplete ones are retried.
    fn json(
        &mut self,
        task: &str,
        system: &str,
        user: &str,
        schema: &Value,
        valid: impl Fn(&Value) -> bool,
    ) -> Result<Option<Value>, String> {
        let schema_text = schema.to_string();
        let digest = fnv1a(&[&self.model, &self.base_url, system, user, &schema_text]);
        let path = self.cache_dir.join(format!("{task}-{digest:016x}.json"));
        if let Ok(text) = fs::read_to_string(&path)
            && let Ok(v) = serde_json::from_str::<Value>(&text)
            && valid(&v)
        {
            self.cached += 1;
            return Ok(Some(v));
        }
        self.calls += 1;
        if self.dry_run {
            return Ok(None);
        }
        let key = self.api_key.clone().ok_or("set OPENAI_API_KEY to call the LLM API")?;
        let body = json!({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": task, "strict": true, "schema": schema}},
        });
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let auth = format!("Bearer {key}");
        let headers = [("Authorization", auth.as_str()), ("Content-Type", "application/json")];
        let mut last_err = String::new();
        for _ in 0..3 {
            let resp = self.client.post(&url, &headers, body.to_string().as_bytes()).map_err(|e| redact(&e))?;
            let text = String::from_utf8_lossy(&resp.body);
            if resp.status != 200 {
                return Err(redact(&format!("{url}: HTTP {}: {}", resp.status, text.chars().take(400).collect::<String>())));
            }
            let reply: Value = serde_json::from_str(&text).map_err(|e| format!("{url}: invalid response: {e}"))?;
            if let Some(usage) = reply.get("usage") {
                self.prompt_tokens += usage["prompt_tokens"].as_u64().unwrap_or(0);
                self.completion_tokens += usage["completion_tokens"].as_u64().unwrap_or(0);
            }
            let content = reply["choices"][0]["message"]["content"].as_str().unwrap_or_default();
            match serde_json::from_str::<Value>(content) {
                Ok(v) if valid(&v) => {
                    fs::create_dir_all(&self.cache_dir).map_err(|e| e.to_string())?;
                    fs::write(&path, v.to_string()).map_err(|e| format!("{}: {e}", path.display()))?;
                    return Ok(Some(v));
                }
                Ok(_) => last_err = format!("{task}: model reply incomplete: {}", content.chars().take(200).collect::<String>()),
                Err(e) => last_err = format!("{task}: model returned invalid JSON ({e}): {}", content.chars().take(200).collect::<String>()),
            }
        }
        Err(last_err)
    }
}

/// FNV-1a over NUL-separated parts: a cache key that is stable across Rust releases.
fn fnv1a(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for b in part.bytes().chain([0]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Mask anything that looks like an API key (`sk-…`) in error text.
fn redact(text: &str) -> String {
    text.split(' ')
        .map(|w| if w.trim_start_matches(['"', '\'']).starts_with("sk-") { "sk-***" } else { w })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Default)]
pub struct RefineReport {
    pub kept: usize,
    pub renamed: usize,
    pub dropped: usize,
    pub merged: usize,
    pub undecided: usize,
    /// Concepts proposed by discovery, and the ones added with their variants.
    pub discover_proposed: usize,
    pub discovered: usize,
    pub discovered_variants: usize,
    /// Discovered variants rejected as known, unattested or too broad.
    pub discover_known: usize,
    pub discover_unattested: usize,
    pub discover_too_broad: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub unattested: usize,
    pub conflicting: usize,
    pub duplicates: usize,
    /// Single ordinary words proposed for multi-word concepts.
    pub too_broad: usize,
    /// Average document length with the reviewed lexicon (written to stats.toml).
    pub avg_len: f32,
}

pub fn refine<T: Transport, C: Clock>(
    llm: &mut Llm<T, C>,
    cache: &Cache,
    filter: &Filter,
    out: &Path,
    cfg: &RefineConfig,
) -> Result<RefineReport, String> {
    let read = |file: &str| -> Result<Vec<TermDef>, String> {
        let path = out.join(file);
        let Ok(src) = fs::read_to_string(&path) else { return Ok(Vec::new()) };
        let parsed: LexiconFile = toml::from_str(&src).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(parsed.term)
    };
    if cfg.review_batch == 0 || cfg.expand_batch == 0 || cfg.discover_batch == 0 {
        return Err("batch sizes must be at least 1".into());
    }
    let curated = read("curated.toml")?;
    let mined = read("mined.toml")?;
    if mined.is_empty() {
        return Err(format!("no mined.toml in {}; run `scitok-learn build` first", out.display()));
    }
    let docs: Vec<String> = cache.records()?.iter().filter(|r| filter.accepts(r)).map(document).collect();
    if docs.is_empty() {
        return Err(format!("no cached {} records in {}..{}", filter.category, filter.from, filter.until));
    }
    let corpus = Corpus::new(docs);
    let category = &filter.category;
    let mut report = RefineReport::default();

    // 1. review mined entries
    let occ = corpus.scan(&mined, 2, &[]);
    let curated_names: HashSet<&str> = curated.iter().map(|d| d.canonical.as_str()).collect();
    let mined_names: HashSet<&str> = mined.iter().map(|d| d.canonical.as_str()).collect();
    let mut decisions: Vec<Option<Decision>> = vec![None; mined.len()];
    let targets = curated.iter().map(|d| format!("{}: {}", d.canonical, d.variants.join("; "))).collect::<Vec<_>>().join("\n");
    for start in (0..mined.len()).step_by(cfg.review_batch) {
        let end = (start + cfg.review_batch).min(mined.len());
        let mut user = format!("Curated concepts (possible merge targets), as `canonical: variants`:\n{targets}\n\nEntries to review:\n");
        for (i, d) in mined.iter().enumerate().take(end).skip(start) {
            write!(user, "id {i} | canonical: {} | variants: {} | documents: {}", d.canonical, d.variants.join("; "), occ.df[i]).unwrap();
            for s in &occ.snippets[i] {
                write!(user, " | context: \"…{s}…\"").unwrap();
            }
            user.push('\n');
        }
        // a reply must decide every id of the batch
        let covers_batch = |v: &Value| {
            let ids: HashSet<u64> = v["decisions"].as_array().into_iter().flatten().filter_map(|d| d["id"].as_u64()).collect();
            (start..end).all(|i| ids.contains(&(i as u64)))
        };
        let Some(v) = llm.json("review", &review_prompt(category), &user, &review_schema(), covers_batch)? else { continue };
        for item in v["decisions"].as_array().into_iter().flatten() {
            let Some(id) = item["id"].as_u64().map(|i| i as usize).filter(|i| (start..end).contains(i)) else { continue };
            decisions[id] = Some(Decision {
                kind: item["decision"].as_str().unwrap_or("keep").to_owned(),
                merge_into: item["merge_into"].as_str().map(|t| snake_case(t.trim().trim_start_matches('~'))),
                name: item["name"].as_str().map(str::to_owned),
                reason: item["reason"].as_str().unwrap_or_default().to_owned(),
            });
        }
    }

    // Final canonical of every kept entry (after renames), so merges can be
    // resolved through renames, drops and chains of merges.
    let mut taken: HashSet<String> = curated_names.iter().chain(mined_names.iter()).map(|s| s.to_string()).collect();
    let mut final_name: Vec<Option<String>> = vec![None; mined.len()];
    for (i, def) in mined.iter().enumerate() {
        match &decisions[i] {
            Some(d) if d.kind == "drop" || d.kind == "merge" => {}
            d => {
                let renamed = d
                    .as_ref()
                    .and_then(|d| d.name.as_deref())
                    .map(snake_case)
                    .filter(|n| !n.is_empty() && *n != def.canonical && taken.insert(n.clone()));
                final_name[i] = Some(renamed.unwrap_or_else(|| def.canonical.clone()));
            }
        }
    }
    let index: HashMap<&str, usize> = mined.iter().enumerate().map(|(i, d)| (d.canonical.as_str(), i)).collect();
    // Follow merge targets until a curated or kept concept; None for cycles,
    // drops and unknown names.
    let resolve = |i: usize| -> Option<String> {
        let mut visited = HashSet::from([i]);
        let mut target = decisions[i].as_ref()?.merge_into.clone()?;
        loop {
            if curated_names.contains(target.as_str()) {
                return Some(target);
            }
            let j = *index.get(target.as_str())?;
            if !visited.insert(j) {
                return None;
            }
            if let Some(name) = &final_name[j] {
                return Some(name.clone());
            }
            match &decisions[j] {
                Some(d) if d.kind == "merge" => target = d.merge_into.clone()?,
                _ => return None,
            }
        }
    };

    let mut kept: Vec<(TermDef, String)> = Vec::new();
    let mut merged: Vec<(TermDef, String)> = Vec::new();
    let mut dropped: Vec<(TermDef, String)> = Vec::new();
    for (i, def) in mined.iter().enumerate() {
        let df = occ.df[i];
        let d = decisions[i].as_ref();
        let reason = d.map(|d| d.reason.as_str()).unwrap_or_default();
        match d.map(|d| d.kind.as_str()) {
            Some("drop") => {
                report.dropped += 1;
                dropped.push((def.clone(), reason.to_owned()));
            }
            Some("merge") => match resolve(i) {
                Some(target) => {
                    report.merged += 1;
                    let note = format!("llm: merged from {}, df={df}, {reason}", def.canonical);
                    merged.push((TermDef { canonical: target, ..def.clone() }, note));
                }
                None => {
                    report.kept += 1;
                    kept.push((def.clone(), format!("llm: merge target unavailable (kept), df={df}, {reason}")));
                }
            },
            None => {
                report.undecided += 1;
                kept.push((def.clone(), format!("llm: undecided (kept), df={df}")));
            }
            Some(_) => {
                let name = final_name[i].clone().unwrap();
                if name != def.canonical {
                    report.renamed += 1;
                }
                report.kept += 1;
                kept.push((TermDef { canonical: name, ..def.clone() }, format!("llm: keep, df={df}, {reason}")));
            }
        }
    }

    // the lexicon so far; proposals matching it are duplicates or conflicts
    // (dropped phrases are included under a placeholder name, so they cannot
    // come back)
    let mut existing = Lexicon::default();
    for d in curated.iter().chain(kept.iter().map(|(d, _)| d)).chain(merged.iter().map(|(d, _)| d)) {
        existing.add(d);
    }
    for (d, _) in &dropped {
        existing.add(&TermDef { canonical: DROPPED.into(), ..d.clone() });
    }

    // 2. discover concepts the miner missed
    let discovered = if cfg.discover_docs > 0 { discover(llm, &corpus, &existing, &taken, category, cfg, &mut report)? } else { Added::new() };
    for ((canonical, case_sensitive), vs) in &discovered {
        existing.add(&TermDef { canonical: canonical.clone(), variants: vs.iter().map(|(v, _)| v.clone()).collect(), case_sensitive: *case_sensitive });
    }

    // 3. expand variants of curated, kept and discovered concepts
    let mut concepts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for d in curated.iter().chain(kept.iter().map(|(d, _)| d)).chain(merged.iter().map(|(d, _)| d)) {
        concepts.entry(d.canonical.clone()).or_default().extend(d.variants.iter().cloned());
    }
    // discovered concepts go last, so earlier batches (and their cache entries) stay the same
    let mut names: Vec<String> = concepts.keys().cloned().collect();
    for ((canonical, _), vs) in &discovered {
        if !concepts.contains_key(canonical) {
            names.push(canonical.clone());
        }
        concepts.entry(canonical.clone()).or_default().extend(vs.iter().map(|(v, _)| v.clone()));
    }
    let mut proposals: Vec<(String, String)> = Vec::new();
    for chunk in names.chunks(cfg.expand_batch) {
        let mut user = String::from("Concepts, as `canonical: existing variants`:\n");
        for c in chunk {
            writeln!(user, "{c}: {}", concepts[c].join("; ")).unwrap();
        }
        let has_concepts = |v: &Value| v["concepts"].is_array();
        let Some(v) = llm.json("expand", &expand_prompt(category), &user, &concepts_schema(), has_concepts)? else { continue };
        for item in v["concepts"].as_array().into_iter().flatten() {
            let Some(c) = item["canonical"].as_str().filter(|c| chunk.iter().any(|n| n == c)) else { continue };
            for var in item["variants"].as_array().into_iter().flatten().filter_map(Value::as_str).take(8) {
                proposals.push((c.to_owned(), var.trim().to_owned()));
            }
        }
    }
    report.proposed = proposals.len();

    // concepts whose variants are all single words may gain single-word
    // variants (spellings); others may not, since a single ordinary word such
    // as "tomography" would replace that word everywhere
    let single_word_concepts: HashSet<&String> = concepts
        .iter()
        .filter(|(_, vs)| vs.iter().all(|v| lex(&normalize(v)).iter().filter(|a| a.is_term()).count() == 1))
        .map(|(c, _)| c)
        .collect();
    let mut buf = String::new();
    let mut by_key: HashMap<String, HashSet<String>> = HashMap::new();
    let mut candidates: Vec<(String, String, String)> = Vec::new();
    for (canonical, variant) in proposals {
        match classify(&existing, &variant, &mut buf) {
            None => {}
            Some(Variant::Known(owner)) if owner == canonical => report.duplicates += 1,
            Some(Variant::Known(_)) => report.conflicting += 1,
            Some(Variant::New { atoms: 1, acronym: false, .. }) if !single_word_concepts.contains(&canonical) => report.too_broad += 1,
            Some(Variant::New { key, .. }) => {
                by_key.entry(key.clone()).or_default().insert(canonical.clone());
                candidates.push((canonical, variant, key));
            }
        }
    }
    // the same new variant proposed for two concepts is ambiguous
    let before = candidates.len();
    candidates.retain(|(_, _, key)| by_key[key].len() == 1);
    report.conflicting += before - candidates.len();
    // attestation in the corpus, and case sensitivity for acronyms that are also words
    let defs: Vec<TermDef> = candidates
        .iter()
        // acronyms are attested with their exact letter case
        .map(|(_, v, _)| TermDef { canonical: String::new(), variants: vec![v.clone()], case_sensitive: single_acronym(v).is_some() })
        .collect();
    let acronym_words: Vec<String> = candidates.iter().filter_map(|(_, v, _)| single_acronym(v)).collect();
    let attest = corpus.scan(&defs, 0, &acronym_words);
    let mut added = Added::new();
    let mut seen = HashSet::new();
    for (i, (canonical, variant, _)) in candidates.iter().enumerate() {
        if attest.df[i] < cfg.min_attested {
            report.unattested += 1;
            continue;
        }
        if !seen.insert((canonical.clone(), variant.to_lowercase())) {
            continue;
        }
        let case_sensitive = single_acronym(variant).is_some_and(|w| attest.word_counts.get(&w).copied().unwrap_or(0) >= cfg.min_attested);
        added.entry((canonical.clone(), case_sensitive)).or_default().push((variant.clone(), attest.df[i]));
        report.accepted += 1;
    }

    if llm.dry_run {
        return Ok(report);
    }
    let dropped: Vec<(String, String)> = dropped.into_iter().map(|(d, r)| (d.canonical, r)).collect();
    let text = reviewed_toml(category, &kept, &merged, &discovered, &added, &dropped, &llm.model);
    // never write a file that would fail to load (it may be compiled into the library)
    let domain = scitok::Domain::from_sources(category, read_text(out, "curated.toml").as_deref(), Some(&text), None, None)
        .map_err(|e| format!("generated reviewed.toml does not load: {e}"))?;
    let path = out.join("reviewed.toml");
    fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;

    // BM25 average length changes with the lexicon
    let tok = Tokenizer::new(Config { domains: vec![domain], ..Config::default() });
    let total: usize = corpus.docs.iter().map(|d| tok.len_as(d, Input::Auto)).sum();
    report.avg_len = (total as f32 / corpus.docs.len() as f32 * 10.0).round() / 10.0;
    let stats_path = out.join("stats.toml");
    if let Some(src) = read_text(out, "stats.toml") {
        let mut stats: scitok::Stats = toml::from_str(&src).map_err(|e| format!("{}: {e}", stats_path.display()))?;
        stats.avg_len = report.avg_len;
        let text = format!("# Generated by scitok-learn; do not edit.\n{}", toml::to_string(&stats).unwrap());
        fs::write(&stats_path, text).map_err(|e| format!("{}: {e}", stats_path.display()))?;
    }
    Ok(report)
}

fn read_text(dir: &Path, file: &str) -> Option<String> {
    fs::read_to_string(dir.join(file)).ok()
}

/// Placeholder canonical for dropped phrases in the lexicon of known terms.
const DROPPED: &str = "\u{0}dropped";

/// Variants added per (canonical, case_sensitive), with their document frequency.
type Added = BTreeMap<(String, bool), Vec<(String, u32)>>;

enum Variant {
    /// Already a variant of this concept.
    Known(String),
    New { key: String, atoms: usize, acronym: bool },
}

fn classify(existing: &Lexicon, variant: &str, buf: &mut String) -> Option<Variant> {
    let normalized = normalize(variant);
    let atoms: Vec<Atom> = lex(&normalized).into_iter().filter(Atom::is_term).collect();
    if atoms.is_empty() {
        return None;
    }
    if let Some((n, id)) = existing.longest_match(&atoms, buf)
        && n == atoms.len()
    {
        return Some(Variant::Known(existing.term(id)[1..].to_owned()));
    }
    let key = atoms.iter().map(scitok::lexicon::key).collect::<Vec<_>>().join(" ");
    Some(Variant::New { key, atoms: atoms.len(), acronym: atoms.len() == 1 && atoms[0].kind == Kind::Acronym })
}

/// Ask for domain concepts in a sample of abstracts that the lexicon lacks.
/// Variants proposed under several names go to the most frequent one; a
/// concept is added if it keeps an attested multi-word variant.
fn discover<T: Transport, C: Clock>(
    llm: &mut Llm<T, C>,
    corpus: &Corpus,
    existing: &Lexicon,
    taken: &HashSet<String>,
    category: &str,
    cfg: &RefineConfig,
    report: &mut RefineReport,
) -> Result<Added, String> {
    // a deterministic sample: the documents with the smallest hashes
    let mut sample: Vec<usize> = (0..corpus.docs.len()).collect();
    sample.sort_by_key(|&i| fnv1a(&[&corpus.docs[i]]));
    sample.truncate(cfg.discover_docs);
    let mut buf = String::new();
    // key -> proposed canonical -> votes, and the key's first surface form
    let mut votes: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    let mut surface: HashMap<String, String> = HashMap::new();
    for chunk in sample.chunks(cfg.discover_batch) {
        let mut user = String::new();
        for (k, &i) in chunk.iter().enumerate() {
            let known = corpus.concepts(&corpus.docs[i], existing);
            let known = if known.is_empty() { "-".to_owned() } else { known.join("; ") };
            writeln!(user, "Abstract {k}:\n{}\nAlready in the lexicon: {known}\n", corpus.docs[i]).unwrap();
        }
        let has_concepts = |v: &Value| v["concepts"].is_array();
        let Some(v) = llm.json("discover", &discover_prompt(category), &user, &concepts_schema(), has_concepts)? else { continue };
        for item in v["concepts"].as_array().into_iter().flatten() {
            let canonical = snake_case(item["canonical"].as_str().unwrap_or_default());
            if canonical.is_empty() {
                continue;
            }
            report.discover_proposed += 1;
            for var in item["variants"].as_array().into_iter().flatten().filter_map(Value::as_str).take(8) {
                match classify(existing, var, &mut buf) {
                    None => {}
                    Some(Variant::Known(_)) => report.discover_known += 1,
                    Some(Variant::New { atoms: 1, acronym: false, .. }) => report.discover_too_broad += 1,
                    Some(Variant::New { key, .. }) => {
                        *votes.entry(key.clone()).or_default().entry(canonical.clone()).or_default() += 1;
                        surface.entry(key).or_insert_with(|| var.trim().to_owned());
                    }
                }
            }
        }
    }

    // canonical -> (variant, key); names of existing concepts are not reused
    let mut concepts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (key, names) in &votes {
        let best = names.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).unwrap().0;
        if taken.contains(best) {
            report.discover_known += 1;
        } else {
            concepts.entry(best.clone()).or_default().push(surface[key].clone());
        }
    }
    let variants: Vec<&String> = concepts.values().flatten().collect();
    let defs: Vec<TermDef> = variants
        .iter()
        .map(|v| TermDef { canonical: String::new(), variants: vec![v.to_string()], case_sensitive: single_acronym(v).is_some() })
        .collect();
    let acronym_words: Vec<String> = variants.iter().filter_map(|v| single_acronym(v)).collect();
    let attest = corpus.scan(&defs, 0, &acronym_words);
    let df: HashMap<&String, u32> = variants.iter().copied().zip(attest.df.iter().copied()).collect();

    let mut added = Added::new();
    for (canonical, vs) in &concepts {
        let attested: Vec<&String> = vs.iter().filter(|v| df[v] >= cfg.min_attested).collect();
        report.discover_unattested += vs.len() - attested.len();
        // an acronym alone only renames a word
        if !attested.iter().any(|v| single_acronym(v).is_none()) {
            report.discover_too_broad += attested.len();
            continue;
        }
        report.discovered += 1;
        report.discovered_variants += attested.len();
        for v in attested {
            let case_sensitive = single_acronym(v).is_some_and(|w| attest.word_counts.get(&w).copied().unwrap_or(0) >= cfg.min_attested);
            added.entry((canonical.clone(), case_sensitive)).or_default().push((v.clone(), df[v]));
        }
    }
    Ok(added)
}

#[derive(Clone)]
struct Decision {
    kind: String,
    merge_into: Option<String>,
    name: Option<String>,
    reason: String,
}

fn review_prompt(category: &str) -> String {
    format!(
        "You curate a search lexicon for arXiv {category} papers. Each entry below was mined automatically from \
         abstracts; it becomes one search concept that matches all of its variants. For every entry decide:\n\
         - keep: a genuine domain concept a researcher might search for (a technique, model, algorithm, phenomenon, \
         object, named result or established acronym).\n\
         - drop: generic scientific or academic phrasing, boilerplate, sentence fragments, or word combinations that \
         are not a concept (e.g. 'good agreement', 'numerical simulations', 'recent years').\n\
         - merge: exactly the same concept as one of the curated concepts or another entry; give its canonical name in \
         merge_into. Only for true synonyms or acronym/long-form pairs, never for broader or narrower concepts.\n\
         For kept entries you may give a cleaner canonical name in `name` (lowercase snake_case, singular, e.g. \
         'spin_chain'), or null to keep it. merge_into is null unless decision is merge. Keep reasons short. Return \
         one decision per id."
    )
}

fn review_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false, "required": ["decisions"],
        "properties": {"decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": false,
            "required": ["id", "decision", "merge_into", "name", "reason"],
            "properties": {
                "id": {"type": "integer"},
                "decision": {"type": "string", "enum": ["keep", "drop", "merge"]},
                "merge_into": {"type": ["string", "null"]},
                "name": {"type": ["string", "null"]},
                "reason": {"type": "string"}
            }
        }}}
    })
}

fn expand_prompt(category: &str) -> String {
    format!(
        "You extend a search lexicon for arXiv {category} papers. For each concept, list other ways researchers \
         write exactly the same concept in titles and abstracts that are missing from its variants: synonyms and \
         alternative names, the acronym or its spelled-out form, British/American spellings, and notation written \
         out (e.g. 'controlled-Z gate' for 'CZ gate'). Rules: the same meaning only, never broader, narrower or \
         merely related concepts; skip variants that differ only by plural, possessive, letter case or hyphens versus \
         spaces (those already match); at most 6 per concept. Many concepts need no additions; return an empty list \
         for them. Return every canonical you were given."
    )
}

fn discover_prompt(category: &str) -> String {
    format!(
        "You build a search lexicon for arXiv {category} papers. From each abstract below, list the domain-specific \
         concepts a researcher might search for that are missing from its 'Already in the lexicon' list: named \
         methods, algorithms, protocols, theorems, inequalities, models, codes, devices, materials and phenomena, \
         with their acronyms. Skip generic scientific phrasing (e.g. 'numerical simulation', 'experimental \
         results'), single ordinary words, and names coined for a single paper. For each concept give a canonical \
         name (lowercase snake_case, singular) and its variants exactly as written in the abstract, including both \
         the acronym and its long form when the abstract gives both. Return an empty list if nothing qualifies."
    )
}

fn concepts_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false, "required": ["concepts"],
        "properties": {"concepts": {"type": "array", "items": {
            "type": "object", "additionalProperties": false, "required": ["canonical", "variants"],
            "properties": {"canonical": {"type": "string"}, "variants": {"type": "array", "items": {"type": "string"}}}
        }}}
    })
}

/// `Spin Chains!` → `spin_chains`.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for c in normalize(name).chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_owned()
}

/// Lowercase form of a single-acronym variant (`ETH` → `eth`).
fn single_acronym(variant: &str) -> Option<String> {
    let normalized = normalize(variant);
    let atoms: Vec<Atom> = lex(&normalized).into_iter().filter(Atom::is_term).collect();
    match atoms.as_slice() {
        [a] if a.kind == Kind::Acronym => Some(a.text.to_lowercase()),
        _ => None,
    }
}

struct Corpus {
    docs: Vec<String>,
    tok: Tokenizer,
}

struct Occurrences {
    df: Vec<u32>,
    snippets: Vec<Vec<String>>,
    /// Occurrences of the given lowercase words as plain lowercase words.
    word_counts: HashMap<String, u32>,
}

impl Corpus {
    fn new(docs: Vec<String>) -> Self {
        Corpus { docs, tok: Tokenizer::new(Config { domains: vec![], ..Config::default() }) }
    }

    /// Names of the lexicon's concepts that occur in `doc` (dropped phrases excluded).
    fn concepts(&self, doc: &str, lexicon: &Lexicon) -> Vec<String> {
        let mut found = std::collections::BTreeSet::new();
        let mut buf = String::new();
        self.tok.with_atoms(doc, Input::Auto, |atoms| {
            for i in 0..atoms.len() {
                if let Some((_, id)) = lexicon.longest_match(&atoms[i..], &mut buf)
                    && lexicon.term(id)[1..] != *DROPPED
                {
                    found.insert(lexicon.term(id)[1..].to_owned());
                }
            }
        });
        found.into_iter().collect()
    }

    /// Document frequency and up to `max_snippets` contexts per definition.
    fn scan(&self, defs: &[TermDef], max_snippets: usize, words: &[String]) -> Occurrences {
        let mut lexicon = Lexicon::default();
        for (i, d) in defs.iter().enumerate() {
            lexicon.add(&TermDef { canonical: i.to_string(), ..d.clone() });
        }
        let words: HashSet<&str> = words.iter().map(String::as_str).collect();
        let mut occ = Occurrences { df: vec![0; defs.len()], snippets: vec![Vec::new(); defs.len()], word_counts: HashMap::new() };
        let mut buf = String::new();
        for doc in &self.docs {
            self.tok.with_atoms(doc, Input::Auto, |atoms| {
                let mut seen = HashSet::new();
                for i in 0..atoms.len() {
                    let a = &atoms[i];
                    if a.kind == Kind::Word && words.contains(a.text) {
                        *occ.word_counts.entry(a.text.to_owned()).or_default() += 1;
                    }
                    let Some((n, id)) = lexicon.longest_match(&atoms[i..], &mut buf) else { continue };
                    let def: usize = lexicon.term(id)[1..].parse().unwrap();
                    if seen.insert(def) {
                        occ.df[def] += 1;
                        if occ.snippets[def].len() < max_snippets {
                            let ctx = &atoms[i.saturating_sub(10)..(i + n + 10).min(atoms.len())];
                            occ.snippets[def].push(ctx.iter().filter(|a| a.is_term()).map(|a| a.text).collect::<Vec<_>>().join(" "));
                        }
                    }
                }
            });
        }
        occ
    }
}

fn reviewed_toml(
    category: &str,
    kept: &[(TermDef, String)],
    merged: &[(TermDef, String)],
    discovered: &Added,
    added: &Added,
    dropped: &[(String, String)],
    model: &str,
) -> String {
    let q = |s: &str| toml::Value::String(s.to_owned()).to_string();
    let comment = |s: &str| s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect::<String>();
    let mut out = format!(
        "# LLM-reviewed {category} lexicon, generated by `scitok-learn refine` ({model}) from mined.toml.\n\
         # Loaded instead of mined.toml. Do not edit; override in curated.toml.\nversion = {FORMAT_VERSION}\n"
    );
    let section = |title: &str, defs: &[(TermDef, String)], out: &mut String| {
        if defs.is_empty() {
            return;
        }
        write!(out, "\n# --- {title} ({}) ---\n", defs.len()).unwrap();
        for (d, note) in defs {
            write!(out, "\n[[term]] # {}\ncanonical = {}\nvariants = [", comment(note), q(&d.canonical)).unwrap();
            out.push_str(&d.variants.iter().map(|v| q(v)).collect::<Vec<_>>().join(", "));
            out.push_str("]\n");
            if d.case_sensitive {
                out.push_str("case_sensitive = true\n");
            }
        }
    };
    section("Reviewed mined terms", kept, &mut out);
    section("Merged into other concepts", merged, &mut out);
    let defs = |added: &Added, label: &str| -> Vec<(TermDef, String)> {
        added
            .iter()
            .map(|((canonical, cs), vs)| {
                let note = vs.iter().map(|(v, df)| format!("{v} df={df}")).collect::<Vec<_>>().join("; ");
                (
                    TermDef { canonical: canonical.clone(), variants: vs.iter().map(|(v, _)| v.clone()).collect(), case_sensitive: *cs },
                    format!("llm: {label}, {note}"),
                )
            })
            .collect()
    };
    section("Discovered concepts", &defs(discovered, "discover"), &mut out);
    section("Added variants", &defs(added, "expand"), &mut out);
    if !dropped.is_empty() {
        write!(out, "\n# --- Dropped ({}) ---\n", dropped.len()).unwrap();
        for (c, reason) in dropped {
            writeln!(out, "# {c}: {}", comment(reason)).unwrap();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Record;
    use crate::http::Rate;
    use crate::http::testing::{FakeClock, FakeTransport};
    use chrono::NaiveDate;
    use std::time::Duration;

    fn no_discovery() -> RefineConfig {
        RefineConfig { discover_docs: 0, ..RefineConfig::default() }
    }

    fn completion(content: &Value) -> Vec<u8> {
        json!({"choices": [{"message": {"content": content.to_string()}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            .to_string()
            .into_bytes()
    }

    fn setup(name: &str) -> (PathBuf, Cache, Filter) {
        let root = std::env::temp_dir().join(format!("scitok-refine-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut cache = Cache::open(&root.join("cache"), "quant-ph").unwrap();
        let texts = [
            "Spin chains with a controlled-phase gate show good agreement.",
            "A controlled-phase gate on spin chains.",
            "Long spin chains, again in good agreement with theory.",
        ];
        let records: Vec<Record> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Record {
                id: format!("2610.0000{i}"),
                datestamp: "2026-10-01".into(),
                categories: vec!["quant-ph".into()],
                title: "T".into(),
                abstract_: t.to_string(),
                ..Record::default()
            })
            .collect();
        cache.append(&records).unwrap();
        let out = root.join("domain");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("curated.toml"), "version = 1\n[[term]]\ncanonical = \"cz_gate\"\nvariants = [\"CZ gate\"]\n").unwrap();
        fs::write(
            out.join("stats.toml"),
            "version = 1\ncategory = \"quant-ph\"\nfrom = \"a\"\nuntil = \"b\"\ngenerated_by = \"t\"\ndocs = 3\navg_len = 99.0\n",
        )
        .unwrap();
        fs::write(
            out.join("mined.toml"),
            "version = 1\n[[term]]\ncanonical = \"good_agreement\"\nvariants = [\"good agreement\"]\n\
             [[term]]\ncanonical = \"spin_chains\"\nvariants = [\"spin chains\"]\n",
        )
        .unwrap();
        let filter = Filter {
            category: "quant-ph".into(),
            from: NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
            until: NaiveDate::from_ymd_opt(2026, 10, 15).unwrap(),
            include_cross_lists: false,
            include_revisions: false,
        };
        (root, cache, filter)
    }

    fn llm(t: FakeTransport, clock: FakeClock, cache_dir: PathBuf) -> Llm<FakeTransport, FakeClock> {
        let mut client = Client::with_clock(t, Rate { requests: 1, per: Duration::from_secs(1) }, clock);
        client.verbose = false;
        Llm {
            client,
            base_url: "http://llm/v1".into(),
            model: "test-model".into(),
            api_key: Some("sk-test".into()),
            cache_dir,
            dry_run: false,
            calls: 0,
            cached: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }

    fn responses(t: &mut FakeTransport) {
        t.route("/chat/completions", 200, b"{\"choices\": [{\"message\": {\"content\": \"not json\"}}]}");
        // incomplete: id 1 missing, so it is retried rather than cached
        t.route("/chat/completions", 200, &completion(&json!({"decisions": [
            {"id": 0, "decision": "drop", "merge_into": null, "name": null, "reason": "generic phrase"}
        ]})));
        t.route("/chat/completions", 200, &completion(&json!({"decisions": [
            {"id": 0, "decision": "drop", "merge_into": null, "name": null, "reason": "generic phrase"},
            {"id": 1, "decision": "keep", "merge_into": null, "name": "Spin chain", "reason": "model\r[[term]] system"}
        ]})));
        t.route("/chat/completions", 200, &completion(&json!({"concepts": [
            {"canonical": "cz_gate", "variants": ["controlled-phase gate", "made-up gate name", "spin chains", "phase", "good agreement"]},
            {"canonical": "spin_chain", "variants": ["CZ gate"]}
        ]})));
    }

    #[test]
    fn reviews_and_expands_with_guards() {
        let (root, cache, filter) = setup("main");
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        responses(&mut t);
        let bodies = t.bodies.clone();
        let out = root.join("domain");
        let mut llm = llm(t, clock, root.join("llm"));
        let report = refine(&mut llm, &cache, &filter, &out, &no_discovery()).unwrap();
        // one malformed and one incomplete reply were retried
        assert_eq!(bodies.borrow().len(), 4);
        assert_eq!((report.kept, report.renamed, report.dropped), (1, 1, 1));
        assert_eq!((report.accepted, report.unattested, report.too_broad), (1, 1, 1));
        // "spin chains" belongs to spin_chain, "CZ gate" to cz_gate, "good agreement" was dropped
        assert_eq!(report.conflicting, 3);
        let stats = fs::read_to_string(out.join("stats.toml")).unwrap();
        assert!(!stats.contains("99.0") && stats.contains(&format!("avg_len = {}", report.avg_len)), "{stats}");

        let reviewed = fs::read_to_string(out.join("reviewed.toml")).unwrap();
        assert!(reviewed.contains("canonical = \"spin_chain\""), "{reviewed}");
        assert!(reviewed.contains("variants = [\"controlled-phase gate\"]"), "{reviewed}");
        assert!(reviewed.contains("# good_agreement: generic phrase"), "{reviewed}");
        assert!(!reviewed.contains("made-up"), "{reviewed}");
        let request = String::from_utf8(bodies.borrow()[0].clone()).unwrap();
        assert!(request.contains("documents: 2"), "{request}");

        // the domain now maps both gate spellings to one concept
        let domain = scitok::Domain::load(&out).unwrap();
        let tok = Tokenizer::new(Config { domains: vec![domain], ..Config::default() });
        assert!(tok.tokenize("a controlled-phase gate").contains(&"~cz_gate".to_string()));
        assert!(!tok.tokenize("good agreement").iter().any(|t| t.starts_with('~')));

        // a second run is served from the cache
        let clock = FakeClock::new();
        let t = FakeTransport::new(clock.clone());
        let bodies = t.bodies.clone();
        let mut llm = super::tests::llm(t, clock, root.join("llm"));
        refine(&mut llm, &cache, &filter, &out, &no_discovery()).unwrap();
        assert_eq!((bodies.borrow().len(), llm.cached), (0, 2));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn discovers_missing_concepts_with_guards() {
        let (root, mut cache, filter) = setup("discover");
        let texts = ["Magic state distillation (MSD) lowers the overhead.", "Magic state distillation with MSD factories.", "Shadow tomography of qubits."];
        let records: Vec<Record> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Record {
                id: format!("2610.0001{i}"),
                datestamp: "2026-10-01".into(),
                categories: vec!["quant-ph".into()],
                title: "T".into(),
                abstract_: t.to_string(),
                ..Record::default()
            })
            .collect();
        cache.append(&records).unwrap();
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        t.route("/chat/completions", 200, &completion(&json!({"decisions": [
            {"id": 0, "decision": "drop", "merge_into": null, "name": null, "reason": "generic"},
            {"id": 1, "decision": "keep", "merge_into": null, "name": null, "reason": "model"}
        ]})));
        let concept = |c: &str, vs: &[&str]| json!({"canonical": c, "variants": vs});
        t.route("/chat/completions", 200, &completion(&json!({"concepts": [
            concept("Magic state distillation", &["magic state distillation", "MSD"]),
            concept("msd_only", &["MSD"]),
            concept("shadow_tomography", &["shadow tomography"]),
            concept("spin_chain_model", &["spin chains"]),
            concept("distillation", &["distillation"]),
            concept("cz_gate", &["controlled phase gate"])
        ]})));
        t.route("/chat/completions", 200, &completion(&json!({"concepts": []})));
        let bodies = t.bodies.clone();
        let out = root.join("domain");
        let mut llm = llm(t, clock, root.join("llm"));
        let r = refine(&mut llm, &cache, &filter, &out, &RefineConfig::default()).unwrap();
        assert_eq!(bodies.borrow().len(), 3);
        assert_eq!((r.discover_proposed, r.discovered, r.discovered_variants), (6, 1, 2));
        // spin chains is known, cz_gate is an existing name; shadow tomography occurs once
        assert_eq!((r.discover_known, r.discover_unattested, r.discover_too_broad), (2, 1, 1));

        let discover = String::from_utf8(bodies.borrow()[1].clone()).unwrap();
        assert!(discover.contains("Already in the lexicon: spin_chains") && !discover.contains("dropped"), "{discover}");
        // discovered concepts are expanded too
        let expand = String::from_utf8(bodies.borrow()[2].clone()).unwrap();
        assert!(expand.contains("magic_state_distillation: magic state distillation; MSD"), "{expand}");
        let reviewed = fs::read_to_string(out.join("reviewed.toml")).unwrap();
        assert!(reviewed.contains("# --- Discovered concepts (1) ---"), "{reviewed}");
        assert!(reviewed.contains("canonical = \"magic_state_distillation\"\nvariants = [\"magic state distillation\", \"MSD\"]"), "{reviewed}");

        let tok = Tokenizer::new(Config { domains: vec![scitok::Domain::load(&out).unwrap()], ..Config::default() });
        assert!(tok.tokenize("MSD").contains(&"~magic_state_distillation".to_string()));
        assert!(!tok.tokenize("shadow tomography").iter().any(|t| t.starts_with('~')));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merge_chains_and_cycles_resolve() {
        let (root, cache, filter) = setup("merge");
        let out = root.join("domain");
        fs::write(
            out.join("mined.toml"),
            "version = 1\n[[term]]\ncanonical = \"a\"\nvariants = [\"controlled phase\"]\n\
             [[term]]\ncanonical = \"b\"\nvariants = [\"cphase gate\"]\n\
             [[term]]\ncanonical = \"c\"\nvariants = [\"spin chains\"]\n\
             [[term]]\ncanonical = \"d\"\nvariants = [\"long spin chains\"]\n",
        )
        .unwrap();
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        let merge = |id: u32, into: &str| json!({"id": id, "decision": "merge", "merge_into": into, "name": null, "reason": "same"});
        t.route("/chat/completions", 200, &completion(&json!({"decisions": [
            merge(0, "b"), merge(1, "~CZ_gate"), merge(2, "d"), merge(3, "c")
        ]})));
        t.route("/chat/completions", 200, &completion(&json!({"concepts": []})));
        let mut llm = llm(t, clock, root.join("llm"));
        let report = refine(&mut llm, &cache, &filter, &out, &no_discovery()).unwrap();
        assert_eq!((report.merged, report.kept), (2, 2));
        let reviewed = fs::read_to_string(out.join("reviewed.toml")).unwrap();
        // a -> b -> cz_gate (target normalized); the c <-> d cycle keeps both
        assert!(reviewed.contains("canonical = \"cz_gate\"\nvariants = [\"controlled phase\"]"), "{reviewed}");
        assert!(reviewed.contains("canonical = \"c\"\nvariants = [\"spin chains\"]"), "{reviewed}");
        assert!(reviewed.contains("canonical = \"d\"\nvariants = [\"long spin chains\"]"), "{reviewed}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn redacts_keys() {
        assert_eq!(redact("Incorrect API key provided: sk-proj-abc123. x"), "Incorrect API key provided: sk-*** x");
    }

    #[test]
    fn api_errors_are_reported() {
        let (root, cache, filter) = setup("error");
        let clock = FakeClock::new();
        let mut t = FakeTransport::new(clock.clone());
        t.route("/chat/completions", 401, b"{\"error\": {\"message\": \"bad key\"}}");
        let mut llm = llm(t, clock, root.join("llm"));
        let err = refine(&mut llm, &cache, &filter, &root.join("domain"), &no_discovery()).unwrap_err();
        assert!(err.contains("HTTP 401") && !err.contains("sk-test"), "{err}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snake_case_names() {
        assert_eq!(snake_case("Spin Chains!"), "spin_chains");
        assert_eq!(snake_case("Schrödinger cat state"), "schrodinger_cat_state");
    }
}
