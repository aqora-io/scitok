//! Domain-aware tokenizer for scientific papers, producing sparse vectors for
//! Qdrant collections configured with `modifier: idf`.
//!
//! ```
//! use scitok::{Config, Tokenizer};
//! let tok = Tokenizer::new(Config::default());
//! let terms = tok.tokenize("We run the quantum approximate optimization algorithm (QAOA).");
//! assert!(terms.contains(&"~qaoa".to_string()));
//! let doc = tok.embed_document("Grover's algorithm on a surface code");
//! assert_eq!(doc.indices.len(), doc.values.len());
//! ```

pub mod acronym;
pub mod domain;
pub mod latex;
pub mod lexer;
pub mod lexicon;
pub mod mine;
pub mod normalize;
pub mod sparse;
#[cfg(feature = "python")]
mod python;

use std::cell::RefCell;

use rayon::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use rustc_hash::{FxHashMap, FxHashSet};

pub use domain::{Domain, Error, Stats};
pub use lexicon::TermDef;
pub use sparse::{SparseVector, Weighting, term_id};

use lexer::{Atom, Kind};
use lexicon::{Lexicon, key, push_key};

const COMMON_STOPWORDS: &str = include_str!("../data/common/stopwords.txt");

/// Average document length used for BM25 when neither the config nor the
/// domain statistics provide one.
pub const FALLBACK_AVG_LEN: f32 = 120.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Input {
    /// LaTeX pre-pass when the text contains `$` or `\`, plain otherwise.
    #[default]
    Auto,
    Plain,
    /// A full LaTeX document (preamble and bibliography are dropped).
    Latex,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Merged in order: curated entries of all domains first, then mined
    /// entries; on conflicting variants the earlier domain wins.
    pub domains: Vec<Domain>,
    pub input: Input,
    pub weighting: Weighting,
    pub options: Options,
}

/// Which parts of the pipeline emit terms (mostly for evaluation/ablation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Emit `~concept` terms for lexicon matches.
    pub concepts: bool,
    /// Also emit the words of multi-word lexicon matches.
    pub phrase_unigrams: bool,
    /// Emit joined hyphenated compounds (`two-qubit` → `twoqubit`).
    pub compounds: bool,
    /// Snowball-stem plain words.
    pub stem: bool,
    /// Expand in-document acronym definitions.
    pub expand_acronyms: bool,
    /// Multiplier on the value of multi-word concept terms (phrase matches),
    /// in documents and queries; 0 drops them. Single-word matches (acronyms)
    /// replace their unigram and always count fully.
    pub concept_weight: f32,
}

impl Default for Options {
    fn default() -> Self {
        // Tuned on the quant-ph evaluation (eval/RESULTS.md): joined compounds
        // hurt ranking, and multi-word concepts work best at half weight.
        Options {
            concepts: true,
            phrase_unigrams: true,
            compounds: false,
            stem: true,
            expand_acronyms: true,
            concept_weight: 0.5,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        #[cfg(feature = "domain-quant-ph")]
        let domains = vec![Domain::builtin("quant-ph").expect("built-in quant-ph domain is valid")];
        #[cfg(not(feature = "domain-quant-ph"))]
        let domains = Vec::new();
        Config { domains, input: Input::Auto, weighting: Weighting::default(), options: Options::default() }
    }
}

pub struct Tokenizer {
    lexicon: Lexicon,
    stopwords: FxHashSet<String>,
    stemmer: Stemmer,
    input: Input,
    weighting: Weighting,
    options: Options,
    avg_len: f32,
}

impl Tokenizer {
    pub fn new(config: Config) -> Tokenizer {
        let mut lexicon = Lexicon::default();
        for def in config.domains.iter().flat_map(|d| &d.curated) {
            lexicon.add(def);
        }
        for def in config.domains.iter().flat_map(|d| &d.mined) {
            lexicon.add(def);
        }
        let mut stopwords: FxHashSet<String> = domain::parse_word_list(COMMON_STOPWORDS).into_iter().collect();
        stopwords.extend(config.domains.iter().flat_map(|d| d.stopwords.iter().cloned()));
        let stats = config.domains.iter().find_map(|d| d.stats.as_ref());
        let avg_len = match config.weighting {
            Weighting::Bm25 { avg_len: Some(len), .. } => len,
            _ => stats
                .and_then(|s| if config.input == Input::Latex { s.avg_len_fulltext } else { Some(s.avg_len) })
                .unwrap_or(FALLBACK_AVG_LEN),
        };
        Tokenizer {
            lexicon,
            stopwords,
            stemmer: Stemmer::create(Algorithm::English),
            input: config.input,
            weighting: config.weighting,
            options: config.options,
            avg_len,
        }
    }

    /// Average document length used for BM25 weighting.
    pub fn avg_len(&self) -> f32 {
        self.avg_len
    }

    /// Number of concept terms in the merged lexicon.
    pub fn lexicon_len(&self) -> usize {
        self.lexicon.len()
    }

    pub fn is_stopword(&self, lowercase: &str) -> bool {
        self.stopwords.contains(lowercase)
    }

    /// The terms a text is indexed under, in order (with repeats).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();
        self.terms(text, |t, _| terms.push(t.to_owned()));
        terms
    }

    /// BM25- (or TF-) weighted sparse vector for indexing.
    pub fn embed_document(&self, text: &str) -> SparseVector {
        let mut counter = sparse::Counter::default();
        self.terms(text, |t, phrase| counter.add(t, self.weight(phrase)));
        counter.document(self.weighting, self.avg_len)
    }

    /// Sparse vector for searching: weight 1 per unique term.
    pub fn embed_query(&self, text: &str) -> SparseVector {
        let mut counter = sparse::Counter::default();
        self.terms(text, |t, phrase| counter.add(t, self.weight(phrase)));
        counter.query()
    }

    /// `embed_document` over many texts in parallel.
    pub fn embed_documents<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Vec<SparseVector> {
        texts.par_iter().map(|t| self.embed_document(t.as_ref())).collect()
    }

    /// Run `f` on the atoms of `text` before acronym expansion and lexicon
    /// matching (used for corpus mining).
    pub fn with_atoms<R>(&self, text: &str, input: Input, f: impl FnOnce(&[Atom]) -> R) -> R {
        let prepared = prepare(text, input);
        f(&lexer::lex(&prepared))
    }

    /// Does the lexicon already cover this exact key (atom keys joined by spaces)?
    pub fn lexicon_contains(&self, key: &str) -> bool {
        self.lexicon.contains_key(key)
    }

    /// Count of terms emitted for `text` in the given input mode.
    pub fn len_as(&self, text: &str, input: Input) -> usize {
        let prepared = prepare(text, input);
        let mut n = 0;
        self.emit(lexer::lex(&prepared), |_, _| n += 1);
        n
    }

    fn prepare(&self, text: &str) -> String {
        prepare(text, self.input)
    }

    fn weight(&self, phrase: bool) -> f32 {
        if phrase { self.options.concept_weight } else { 1.0 }
    }

    /// In-document acronym definitions as `(short form, long form)`.
    pub fn acronym_definitions(&self, text: &str) -> Vec<(String, String)> {
        let prepared = self.prepare(text);
        let atoms = lexer::lex(&prepared);
        acronym::definitions(&atoms)
            .into_iter()
            .map(|d| {
                let mut long = String::new();
                for a in &atoms[d.long.clone()] {
                    long.push_str(a.text);
                    long.push(if a.hyphen_next { '-' } else { ' ' });
                }
                long.pop();
                (d.short_text(&atoms), long)
            })
            .collect()
    }

    fn terms(&self, text: &str, f: impl FnMut(&str, bool)) {
        let prepared = self.prepare(text);
        self.emit(lexer::lex(&prepared), f);
    }

    fn emit(&self, atoms: Vec<Atom>, mut f: impl FnMut(&str, bool)) {
        let opts = &self.options;
        let atoms = if opts.expand_acronyms {
            acronym::expand(atoms, |a| self.lexicon.contains_key(&key(a)))
        } else {
            atoms
        };
        let (mut buf, mut lower, mut term) = (String::new(), String::new(), String::new());
        let mut no_match_until = 0;
        for (i, atom) in atoms.iter().enumerate() {
            if !atom.is_term() {
                continue;
            }
            // inside a match: unigrams only if phrase_unigrams
            let mut consumed = i < no_match_until && !opts.phrase_unigrams;
            if opts.concepts
                && i >= no_match_until
                && let Some((n, id)) = self.lexicon.longest_match(&atoms[i..], &mut buf)
            {
                if n == 1 || opts.concept_weight > 0.0 {
                    f(self.lexicon.term(id), n > 1);
                }
                consumed = n == 1 || !opts.phrase_unigrams;
                no_match_until = i + n;
            }
            if !consumed && self.unigram(atom, &mut lower, &mut term) {
                f(&term, false);
            }
            if opts.compounds
                && atom.hyphen_next
                && (i == 0 || !atoms[i - 1].hyphen_next)
                && self.compound(&atoms[i..], &mut lower, &mut term)
            {
                f(&term, false);
            }
        }
    }

    /// Write the unigram term of `atom` to `out`; false if it has none.
    fn unigram(&self, atom: &Atom, lower: &mut String, out: &mut String) -> bool {
        if atom.kind == Kind::Number || atom.text.chars().nth(1).is_none() {
            return false;
        }
        lower.clear();
        lower.extend(atom.text.chars().flat_map(char::to_lowercase));
        if self.stopwords.contains(lower.as_str()) {
            return false;
        }
        out.clear();
        match atom.kind {
            Kind::Word if self.options.stem => self.stem(lower, out),
            Kind::Word => out.push_str(lower),
            _ => push_key(atom, out),
        }
        true
    }

    /// `two-qubit` → `twoqubit`, so it also matches the unhyphenated spelling.
    fn compound(&self, atoms: &[Atom], joined: &mut String, out: &mut String) -> bool {
        let Some(end) = atoms.iter().position(|a| !a.hyphen_next) else { return false };
        let parts = &atoms[..=end];
        joined.clear();
        joined.extend(parts.iter().flat_map(|a| a.text.chars()).filter(|&c| c != '_').flat_map(char::to_lowercase));
        if joined.len() < 3 {
            return false;
        }
        out.clear();
        if self.options.stem && parts.iter().all(|a| a.kind == Kind::Word) {
            self.stem(joined, out);
        } else {
            out.push_str(joined);
        }
        true
    }

    /// Append the stem of `lower` to `out`. Snowball stemming is the most
    /// expensive step; word frequencies are Zipfian, so a per-thread cache
    /// removes most of it.
    fn stem(&self, lower: &str, out: &mut String) {
        thread_local! {
            static STEMS: RefCell<FxHashMap<Box<str>, Box<str>>> = RefCell::default();
        }
        STEMS.with_borrow_mut(|cache| {
            if let Some(stem) = cache.get(lower) {
                out.push_str(stem);
                return;
            }
            if cache.len() >= 200_000 {
                cache.clear();
            }
            let stem = self.stemmer.stem(lower);
            out.push_str(&stem);
            cache.insert(lower.into(), stem.as_ref().into());
        })
    }
}

fn prepare(text: &str, input: Input) -> String {
    match input {
        Input::Plain => normalize::normalize(text),
        Input::Latex => normalize::normalize(&latex::to_text(text, true)),
        Input::Auto if text.contains(['$', '\\']) => {
            normalize::normalize(&latex::to_text(text, text.contains("\\begin{document}")))
        }
        Input::Auto => normalize::normalize(text),
    }
}
