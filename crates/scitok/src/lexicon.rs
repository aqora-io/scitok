//! Phrase/acronym lexicon: every variant maps to one canonical concept term.
//!
//! Variants are lexed with the same lexer as documents and matched on atom
//! keys (lowercase, singularized), so `Lieb-Robinson bounds`, `Lieb Robinson
//! bound` and `Lieb–Robinson bound` are the same key.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::lexer::{Atom, Kind, lex};
use crate::normalize::normalize;

/// Prefix marking concept terms so they never collide with unigrams.
pub const CONCEPT_PREFIX: char = '~';

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct TermDef {
    pub canonical: String,
    pub variants: Vec<String>,
    /// Acronyms in the variants (`ETH`) must match with exact letter case, for
    /// acronyms that are also English words. Other words stay case-insensitive.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub case_sensitive: bool,
}

struct Variant {
    term: u32,
    /// For case-sensitive variants: the singularized surface each acronym atom
    /// must match exactly (`None` for ordinary words).
    surface: Option<Vec<Option<String>>>,
}

#[derive(Default)]
struct Node {
    variants: Vec<Variant>,
    /// Some longer variant starts with this key.
    prefix: bool,
}

#[derive(Default)]
pub struct Lexicon {
    terms: Vec<String>,
    /// Variant keys and their prefixes, so matching costs one lookup per atom.
    nodes: FxHashMap<String, Node>,
}

impl Lexicon {
    /// Add definitions in priority order: when two definitions share a variant
    /// key, the first one added wins.
    pub fn add(&mut self, def: &TermDef) {
        let term = self.terms.len() as u32;
        let mut used = false;
        for variant in &def.variants {
            let normalized = normalize(variant);
            let atoms: Vec<Atom> = lex(&normalized).into_iter().filter(Atom::is_term).collect();
            if atoms.is_empty() {
                continue;
            }
            let keys: Vec<String> = atoms.iter().map(key).collect();
            let surface = def.case_sensitive.then(|| {
                atoms
                    .iter()
                    .map(|a| (a.kind == Kind::Acronym).then(|| singularize(a.text).to_owned()))
                    .collect::<Vec<_>>()
            });
            let node = self.nodes.entry(keys.join(" ")).or_default();
            if node.variants.iter().any(|v| v.surface == surface) {
                continue;
            }
            node.variants.push(Variant { term, surface });
            for n in 1..keys.len() {
                self.nodes.entry(keys[..n].join(" ")).or_default().prefix = true;
            }
            used = true;
        }
        if used {
            self.terms.push(format!("{CONCEPT_PREFIX}{}", def.canonical));
        }
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn term(&self, id: u32) -> &str {
        &self.terms[id as usize]
    }

    /// Does this exact key (one or more atom keys joined by spaces) map to a term?
    pub fn contains_key(&self, key: &str) -> bool {
        self.nodes.get(key).is_some_and(|n| !n.variants.is_empty())
    }

    /// Longest variant starting at `atoms[0]`: `(atoms consumed, term id)`.
    /// `buf` is scratch space for the joined key.
    pub fn longest_match(&self, atoms: &[Atom], buf: &mut String) -> Option<(usize, u32)> {
        buf.clear();
        let mut best = None;
        for (n, atom) in atoms.iter().enumerate() {
            if !atom.is_term() {
                break;
            }
            if n > 0 {
                buf.push(' ');
            }
            push_key(atom, buf);
            let Some(node) = self.nodes.get(buf.as_str()) else { break };
            let span = &atoms[..=n];
            if let Some(v) = node.variants.iter().find(|v| surface_matches(v, span)) {
                best = Some((n + 1, v.term));
            }
            if !node.prefix {
                break;
            }
        }
        best
    }
}

fn surface_matches(v: &Variant, atoms: &[Atom]) -> bool {
    match &v.surface {
        None => true,
        Some(surface) => surface
            .iter()
            .zip(atoms)
            .all(|(s, a)| s.as_deref().is_none_or(|s| s == singularize(a.text))),
    }
}

/// Matching key of an atom: lowercase, `_` removed, singularized.
pub fn key(atom: &Atom) -> String {
    let mut out = String::with_capacity(atom.text.len());
    push_key(atom, &mut out);
    out
}

/// Append the [`key`] of `atom` to `out`.
pub fn push_key(atom: &Atom, out: &mut String) {
    let start = out.len();
    for c in atom.text.chars().filter(|&c| c != '_') {
        if c.is_ascii() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.extend(c.to_lowercase());
        }
    }
    match atom.kind {
        Kind::Group => {}
        // QPUs, CNOTs: a lowercase plural s after capitals
        Kind::Acronym => {
            if singularize(atom.text).len() < atom.text.len() {
                out.pop();
            }
        }
        _ => {
            let word = &out[start..];
            if word.len() > 5 && word.ends_with("ies") {
                out.truncate(out.len() - 3);
                out.push('y');
            } else {
                let len = singularize(word).len();
                out.truncate(start + len);
            }
        }
    }
}

/// Strip a plural `s` (`qubits`, `CNOTs`), keeping `class`, `status`, `analysis`.
pub fn singularize(word: &str) -> &str {
    let b = word.as_bytes();
    if b.len() > 3 && b[b.len() - 1] == b's' && !matches!(b[b.len() - 2], b's' | b'u' | b'i' | b'S') {
        &word[..word.len() - 1]
    } else {
        word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon(defs: &[(&str, &[&str], bool)]) -> Lexicon {
        let mut lex = Lexicon::default();
        for (canonical, variants, cs) in defs {
            lex.add(&TermDef {
                canonical: canonical.to_string(),
                variants: variants.iter().map(|s| s.to_string()).collect(),
                case_sensitive: *cs,
            });
        }
        lex
    }

    fn find(lexicon: &Lexicon, text: &str) -> Option<(usize, String)> {
        let normalized = normalize(text);
        let atoms = lex(&normalized);
        lexicon.longest_match(&atoms, &mut String::new()).map(|(n, t)| (n, lexicon.term(t).to_owned()))
    }

    #[test]
    fn keys() {
        let k = |s: &str| lex(s).iter().map(key).collect::<Vec<_>>().join(" ");
        assert_eq!(k("Lieb-Robinson bounds"), "lieb robinson bound");
        assert_eq!(k("CNOTs QPUs eigenenergies class status analysis Z_2 SU(2)"), "cnot qpu eigenenergy class status analysis z2 su(2)");
    }

    #[test]
    fn longest_match_wins() {
        let lx = lexicon(&[
            ("surface_code", &["surface code"], false),
            ("rotated_surface_code", &["rotated surface code"], false),
            ("lieb_robinson_bound", &["Lieb-Robinson bound"], false),
        ]);
        assert_eq!(find(&lx, "rotated surface codes are"), Some((3, "~rotated_surface_code".into())));
        assert_eq!(find(&lx, "surface code"), Some((2, "~surface_code".into())));
        assert_eq!(find(&lx, "surface"), None);
        assert_eq!(find(&lx, "Lieb–Robinson bounds"), Some((3, "~lieb_robinson_bound".into())));
        assert_eq!(find(&lx, "Lieb Robinson bound"), Some((3, "~lieb_robinson_bound".into())));
        assert_eq!(find(&lx, "surface, code"), None);
    }

    #[test]
    fn case_sensitive_variants() {
        let lx = lexicon(&[("eth", &["ETH", "eigenstate thermalization hypothesis"], true)]);
        assert_eq!(find(&lx, "ETH"), Some((1, "~eth".into())));
        assert_eq!(find(&lx, "eth"), None);
        assert_eq!(find(&lx, "Eigenstate thermalization hypothesis"), Some((3, "~eth".into())));
        let lx = lexicon(&[("qaoa", &["QAOA"], false)]);
        assert_eq!(find(&lx, "qaoa"), Some((1, "~qaoa".into())));
    }

    #[test]
    fn first_definition_wins() {
        let lx = lexicon(&[("a", &["foo bar"], false), ("b", &["foo bars"], false)]);
        assert_eq!(find(&lx, "foo bar"), Some((2, "~a".into())));
        assert_eq!(lx.len(), 1);
    }
}
