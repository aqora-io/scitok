//! Corpus mining for domain lexicons (no IO; driven by `scitok-learn`).
//!
//! Feed documents with [`Miner::add`], then [`Miner::finish`] yields:
//! - acronym ↔ long-form pairs (Schwartz–Hearst, aggregated over the corpus),
//! - collocations (2..=`max_ngram` atoms) ranked by NPMI × ln(document frequency),
//! - eponym patterns (`Lieb-Robinson bound`, `Kochen-Specker theorem`),
//! - candidate stopwords (very high document frequency), for human review.
//!
//! The miner's tokenizer should hold only the curated lexicon, so terms that
//! are already curated are not mined again.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::acronym;
use crate::lexer::{Atom, Kind};
use crate::lexicon::{TermDef, key, singularize};
use crate::{Input, Tokenizer};

const MINING_STOPWORDS: &str = include_str!("../data/common/mining_stopwords.txt");

/// Words that never end a collocation (`van der` Waals).
const NAME_PARTICLES: &[&str] = &["van", "von", "der", "den", "de", "la", "le", "du", "di", "da"];

/// Capitalized prefixes that look like eponyms at the start of a sentence
/// (`Non-Clifford gates`).
const PREFIXES: &[&str] = &[
    "Non", "Anti", "Semi", "Multi", "Quasi", "Pseudo", "Post", "Pre", "Sub", "Super", "Self", "Two",
    "Three", "One", "Many", "Few", "Single", "Long", "Short", "High", "Low",
];

const EPONYM_HEADS: &[&str] = &[
    "bound", "theorem", "inequality", "code", "model", "algorithm", "protocol", "conjecture",
    "lemma", "limit", "relation", "equation", "transformation", "transform", "channel", "state",
    "gate", "effect", "criterion", "entropy", "chain", "decomposition", "game", "sampling",
];

#[derive(Clone, Debug)]
pub struct MineConfig {
    /// Minimum document frequency for acronym pairs and collocations.
    pub min_count: u32,
    /// Minimum document frequency for eponym patterns.
    pub min_eponym_count: u32,
    pub min_npmi: f32,
    pub max_ngram: usize,
    /// Maximum number of collocations kept (best scores first).
    pub max_phrases: usize,
    /// Share of an acronym's definitions that its main long form must have.
    pub acronym_dominance: f32,
    /// Document-frequency ratio above which a word becomes a stopword candidate.
    pub stopword_df: f32,
}

impl Default for MineConfig {
    fn default() -> Self {
        MineConfig {
            min_count: 5,
            min_eponym_count: 2,
            min_npmi: 0.5,
            max_ngram: 4,
            max_phrases: 3000,
            acronym_dominance: 0.6,
            stopword_df: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Acronym,
    Eponym,
    Collocation,
}

#[derive(Clone, Debug)]
pub struct MinedTerm {
    pub def: TermDef,
    pub source: Source,
    /// Documents containing the term (for acronyms: documents defining it).
    pub df: u32,
    /// NPMI for collocations, dominance share for acronyms.
    pub strength: f32,
}

#[derive(Debug, Default)]
pub struct MineResult {
    pub terms: Vec<MinedTerm>,
    /// `(word, document frequency ratio)`, highest first.
    pub stopword_candidates: Vec<(String, f32)>,
}

pub struct Miner<'t> {
    tok: &'t Tokenizer,
    cfg: MineConfig,
    docs: u32,
    /// Counts of all n-grams (n ≤ max_ngram) by hash, for NPMI.
    counts: HashMap<u64, u32>,
    positions: u64,
    /// Document frequency of collocation candidates by hash.
    cand_df: HashMap<u64, u32>,
    /// Keys and a lowercase surface form of candidates that reached
    /// `min_count` documents.
    cand_keys: HashMap<u64, (Vec<String>, String)>,
    /// eponym keys → (documents, lowercase surface form)
    eponym_df: HashMap<Vec<String>, (u32, String)>,
    /// short form → long-form key → (documents, a surface form)
    acronyms: HashMap<String, HashMap<String, (u32, String)>>,
    /// lowercase words seen as ordinary words (for acronym case sensitivity)
    lower_words: HashMap<String, u32>,
    word_df: HashMap<String, u32>,
    mining_stopwords: HashSet<String>,
}

impl<'t> Miner<'t> {
    pub fn new(tok: &'t Tokenizer, cfg: MineConfig) -> Self {
        Miner {
            tok,
            cfg,
            docs: 0,
            counts: HashMap::new(),
            positions: 0,
            cand_df: HashMap::new(),
            cand_keys: HashMap::new(),
            eponym_df: HashMap::new(),
            acronyms: HashMap::new(),
            lower_words: HashMap::new(),
            word_df: HashMap::new(),
            mining_stopwords: crate::domain::parse_word_list(MINING_STOPWORDS).into_iter().collect(),
        }
    }

    /// Add a document. Collocations are only mined from `collocations: true`
    /// documents (abstracts); full texts would need far more memory and
    /// contribute acronyms, eponyms and word statistics instead.
    pub fn add(&mut self, text: &str, input: Input, collocations: bool) {
        let tok = self.tok;
        tok.with_atoms(text, input, |atoms| self.add_atoms(atoms, collocations));
    }

    fn add_atoms(&mut self, atoms: &[Atom], collocations: bool) {
        self.docs += 1;
        let keys: Vec<String> = atoms.iter().map(key).collect();
        let mut seen_words = HashSet::new();
        let mut seen_cands = HashSet::new();
        let mut seen_eponyms = HashSet::new();
        let mut seen_acronyms = HashSet::new();

        for (i, atom) in atoms.iter().enumerate() {
            if atom.kind == Kind::Word && atom.text.chars().all(char::is_lowercase) {
                *self.lower_words.entry(atom.text.to_owned()).or_default() += 1;
            }
            if atom.kind == Kind::Word && seen_words.insert(keys[i].as_str()) {
                *self.word_df.entry(keys[i].clone()).or_default() += 1;
            }
            let chain_start = i == 0 || !atoms[i - 1].hyphen_next;
            if let Some(eponym) = eponym_at(&atoms[i..], &keys[i..]).filter(|_| chain_start)
                && seen_eponyms.insert(eponym.clone()) {
                    let n = eponym.len();
                    self.eponym_df.entry(eponym).or_insert_with(|| (0, surface(&atoms[i..i + n]))).0 += 1;
                }
        }

        for def in acronym::definitions(atoms) {
            let short = singularize(atoms[def.short].text).to_owned();
            let long_key = keys[def.long.clone()].join(" ");
            if !seen_acronyms.insert((short.clone(), long_key.clone())) {
                continue;
            }
            let surface = atoms[def.long.clone()].iter().map(|a| a.text).collect::<Vec<_>>().join(" ");
            let entry = self.acronyms.entry(short).or_default().entry(long_key).or_insert((0, surface));
            entry.0 += 1;
        }

        if !collocations {
            return;
        }
        // spans of consecutive mineable atoms
        let mut start = 0;
        while start < atoms.len() {
            if !mineable(&atoms[start]) {
                start += 1;
                continue;
            }
            let mut end = start;
            while end < atoms.len() && mineable(&atoms[end]) {
                end += 1;
            }
            self.add_span(&atoms[start..end], &keys[start..end], &mut seen_cands);
            start = end;
        }
    }

    fn add_span(&mut self, atoms: &[Atom], keys: &[String], seen: &mut HashSet<u64>) {
        self.positions += atoms.len() as u64;
        for i in 0..atoms.len() {
            let mut hasher = DefaultHasher::new();
            for n in 1..=self.cfg.max_ngram.min(atoms.len() - i) {
                if n > 1 {
                    ' '.hash(&mut hasher);
                }
                keys[i + n - 1].hash(&mut hasher);
                let h = hasher.finish();
                *self.counts.entry(h).or_default() += 1;
                // never cut a hyphenated compound (`no-go theorem` → `go theorem`)
                let whole = (i == 0 || !atoms[i - 1].hyphen_next) && !atoms[i + n - 1].hyphen_next;
                if n >= 2 && whole && self.is_candidate(&atoms[i..i + n]) && seen.insert(h) {
                    let df = self.cand_df.entry(h).or_default();
                    *df += 1;
                    if *df == self.cfg.min_count {
                        self.cand_keys.insert(h, (keys[i..i + n].to_vec(), surface(&atoms[i..i + n])));
                    }
                }
            }
        }
    }

    fn is_candidate(&self, atoms: &[Atom]) -> bool {
        let lower: Vec<String> = atoms.iter().map(|a| a.text.to_lowercase()).collect();
        let stop = |w: &String| self.tok.is_stopword(w);
        let last = &lower[lower.len() - 1];
        atoms.iter().all(|a| a.kind != Kind::Number && a.text.chars().nth(1).is_some())
            && !stop(&lower[0])
            && !stop(last)
            && !NAME_PARTICLES.contains(&last.as_str())
            && lower[1..lower.len() - 1].iter().all(|w| !stop(w) || w == "of")
            && !lower.iter().any(|w| self.mining_stopwords.contains(w))
    }

    fn count(&self, keys: &[String]) -> u32 {
        let mut hasher = DefaultHasher::new();
        for (n, k) in keys.iter().enumerate() {
            if n > 0 {
                ' '.hash(&mut hasher);
            }
            k.hash(&mut hasher);
        }
        self.counts.get(&hasher.finish()).copied().unwrap_or(0)
    }

    /// Minimum NPMI over all binary splits of the n-gram.
    fn npmi(&self, keys: &[String]) -> f32 {
        let total = self.positions.max(1) as f64;
        let p = |keys: &[String]| self.count(keys) as f64 / total;
        let pg = p(keys);
        if pg <= 0.0 || pg >= 1.0 {
            return -1.0;
        }
        (1..keys.len())
            .map(|s| {
                let (l, r) = (p(&keys[..s]), p(&keys[s..]));
                ((pg / (l * r)).ln() / -pg.ln()) as f32
            })
            .fold(f32::INFINITY, f32::min)
    }

    pub fn finish(self) -> MineResult {
        let cfg = &self.cfg;
        let mut terms = Vec::new();
        let mut taken: HashSet<String> = HashSet::new();
        let known = |key: &str, taken: &HashSet<String>| self.tok.lexicon_contains(key) || taken.contains(key);

        // acronyms
        let mut shorts: Vec<_> = self.acronyms.iter().collect();
        shorts.sort_by(|a, b| a.0.cmp(b.0));
        for (short, longs) in shorts {
            let total: u32 = longs.values().map(|(n, _)| n).sum();
            let mut longs: Vec<_> = longs.iter().collect();
            longs.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(b.0)));
            let (main_key, (main_n, _)) = longs[0];
            let share = *main_n as f32 / total as f32;
            let short_key = short.to_lowercase();
            if *main_n < cfg.min_count || share < cfg.acronym_dominance || known(&short_key, &taken) || known(main_key, &taken) {
                continue;
            }
            let mut variants = vec![short.clone()];
            for (long_key, (n, _)) in &longs {
                if *n >= cfg.min_count && !known(long_key, &taken) {
                    variants.push((*long_key).clone());
                    taken.insert((*long_key).clone());
                }
            }
            taken.insert(short_key.clone());
            let case_sensitive = self.lower_words.get(&short_key).copied().unwrap_or(0) >= cfg.min_count;
            terms.push(MinedTerm {
                def: TermDef { canonical: short_key, variants, case_sensitive },
                source: Source::Acronym,
                df: *main_n,
                strength: share,
            });
        }

        // eponyms
        let mut eponyms: Vec<_> = self.eponym_df.iter().filter(|(_, (df, _))| *df >= cfg.min_eponym_count).collect();
        eponyms.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(b.0)));
        for (keys, (df, surface)) in eponyms {
            let joined = keys.join(" ");
            if known(&joined, &taken) {
                continue;
            }
            taken.insert(joined);
            terms.push(MinedTerm {
                def: TermDef { canonical: surface.replace(' ', "_"), variants: vec![surface.clone()], case_sensitive: false },
                source: Source::Eponym,
                df: *df,
                strength: 1.0,
            });
        }

        // collocations: first drop n-grams that (almost) only occur inside a
        // longer frequent n-gram, including ones that are already known terms
        let df_of: HashMap<&[String], u32> = self.cand_keys.iter().map(|(h, (k, _))| (k.as_slice(), self.cand_df[h])).collect();
        let subsumed = |keys: &[String], df: u32| {
            df_of.iter().any(|(longer, ldf)| {
                longer.len() > keys.len()
                    && longer.windows(keys.len()).any(|w| w == keys)
                    && *ldf as f32 >= 0.9 * df as f32
            })
        };
        let mut cands: Vec<(&Vec<String>, &String, u32, f32)> = self
            .cand_keys
            .iter()
            .map(|(h, (keys, surface))| (keys, surface, self.cand_df[h]))
            .filter(|(keys, _, df)| !subsumed(keys, *df) && !known(&keys.join(" "), &taken))
            .map(|(keys, surface, df)| (keys, surface, df, self.npmi(keys)))
            .filter(|c| c.3 >= cfg.min_npmi)
            .collect();
        let score = |df: u32, npmi: f32| npmi * (df as f32).ln();
        cands.sort_by(|a, b| score(b.2, b.3).total_cmp(&score(a.2, a.3)).then(a.0.cmp(b.0)));
        for (_, surface, df, npmi) in cands.into_iter().take(cfg.max_phrases) {
            terms.push(MinedTerm {
                def: TermDef { canonical: surface.replace(' ', "_"), variants: vec![surface.clone()], case_sensitive: false },
                source: Source::Collocation,
                df,
                strength: npmi,
            });
        }

        let docs = self.docs.max(1) as f32;
        let mut stopword_candidates: Vec<(String, f32)> = self
            .word_df
            .iter()
            .map(|(w, df)| (w.clone(), *df as f32 / docs))
            .filter(|(w, ratio)| *ratio >= cfg.stopword_df && !self.tok.is_stopword(w))
            .collect();
        stopword_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        MineResult { terms, stopword_candidates }
    }
}

/// Lowercase surface form, e.g. `lieb robinson bound`.
fn surface(atoms: &[Atom]) -> String {
    atoms.iter().map(|a| a.text.to_lowercase().replace('_', "")).collect::<Vec<_>>().join(" ")
}

fn mineable(a: &Atom) -> bool {
    matches!(a.kind, Kind::Word | Kind::Acronym | Kind::AlphaNum | Kind::Group)
}

/// `Lieb-Robinson bound`, `Greenberger-Horne-Zeilinger state`: a chain of
/// capitalized words joined by hyphens, starting at `atoms[0]`, followed by a
/// lowercase head noun (title-cased `Error-Correcting Codes` is not an eponym).
fn eponym_at(atoms: &[Atom], keys: &[String]) -> Option<Vec<String>> {
    let capitalized = |a: &Atom| {
        let mut cs = a.text.chars();
        a.kind == Kind::Word && cs.next().is_some_and(char::is_uppercase) && cs.clone().count() >= 2 && cs.all(char::is_lowercase)
    };
    if PREFIXES.contains(&atoms.first()?.text) {
        return None;
    }
    let names = atoms.iter().take_while(|a| capitalized(a)).position(|a| !a.hyphen_next)? + 1;
    let head = atoms.get(names)?;
    let lowercase_head = head.kind == Kind::Word && head.text.starts_with(|c: char| c.is_lowercase());
    (names >= 2 && lowercase_head && EPONYM_HEADS.contains(&keys[names].as_str())).then(|| keys[..=names].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Domain};

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(Config { domains: vec![Domain::default()], ..Config::default() })
    }

    fn mine(docs: &[&str], cfg: MineConfig) -> MineResult {
        let tok = tokenizer();
        let mut m = Miner::new(&tok, cfg);
        for d in docs {
            m.add(d, Input::Plain, true);
        }
        m.finish()
    }

    fn cfg() -> MineConfig {
        MineConfig { min_count: 2, ..MineConfig::default() }
    }

    #[test]
    fn mines_acronyms() {
        let r = mine(
            &[
                "We use the variational quantum eigensolver (VQE) here.",
                "A variational quantum eigensolver (VQE) for chemistry.",
                "Training the variational quantum eigensolvers (VQEs).",
            ],
            cfg(),
        );
        let t = r.terms.iter().find(|t| t.source == Source::Acronym).unwrap();
        assert_eq!(t.def.canonical, "vqe");
        assert_eq!(t.def.variants, vec!["VQE", "variational quantum eigensolver"]);
        assert_eq!(t.df, 3);
        assert!(!t.def.case_sensitive);
    }

    #[test]
    fn ambiguous_acronyms_are_dropped() {
        let r = mine(
            &[
                "quantum error correction (QEC) a", "quantum error correction (QEC) b",
                "quantum electronic circuits (QEC) c", "quantum electronic circuits (QEC) d",
            ],
            cfg(),
        );
        assert!(r.terms.iter().all(|t| t.source != Source::Acronym));
    }

    #[test]
    fn mines_eponyms_and_collocations() {
        let docs: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "Doc {i} studies the Lieb-Robinson bound for spin chains and the Greenberger-Horne-Zeilinger state. \\
                     Error-Correcting Codes. Filler word{i} appears alone. No-go theorems. In recent years. \\
                     Non-Clifford gates."
                )
            })
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let r = mine(&refs, cfg());
        let canon: Vec<&str> = r.terms.iter().map(|t| t.def.canonical.as_str()).collect();
        assert!(canon.contains(&"lieb_robinson_bound"), "{canon:?}");
        assert!(canon.contains(&"greenberger_horne_zeilinger_state"), "{canon:?}");
        assert!(!canon.iter().any(|c| c.contains("horne_zeilinger_state") && !c.starts_with("greenberger")), "{canon:?}");
        let eponym = |c: &str| r.terms.iter().any(|t| t.source == Source::Eponym && t.def.canonical == c);
        assert!(eponym("greenberger_horne_zeilinger_state") && !eponym("error_correcting_codes"));
        assert!(!eponym("non_clifford_gates"));
        assert!(canon.contains(&"spin_chains"), "{canon:?}");
        assert!(!canon.iter().any(|c| c.contains("doc")), "{canon:?}");
        assert!(!canon.iter().any(|c| c.starts_with("go_") || c.contains("recent")), "{canon:?}");
        // every document has "studies", "spin", ...: stopword candidates
        assert!(r.stopword_candidates.iter().any(|(w, _)| w == "spin"));
    }
}
