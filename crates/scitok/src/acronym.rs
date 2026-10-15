//! In-document acronym definitions, `long form (LF)`, detected with the
//! Schwartz–Hearst algorithm. Later uses of `LF` in the same document are
//! expanded so they count toward the long form's terms as well.

use crate::lexer::{Atom, BREAK, Kind};
use crate::lexicon::singularize;

pub struct Definition {
    /// Index of the acronym atom (inside the parentheses).
    pub short: usize,
    /// Atom range of the long form.
    pub long: std::ops::Range<usize>,
}

pub fn definitions(atoms: &[Atom]) -> Vec<Definition> {
    let mut defs = Vec::new();
    for i in 1..atoms.len().saturating_sub(2) {
        let short = &atoms[i + 1];
        if atoms[i].kind != Kind::Open || atoms[i + 2].kind != Kind::Close || short.kind != Kind::Acronym {
            continue;
        }
        let letters = singularize(short.text);
        let n = letters.chars().count();
        if !(2..=10).contains(&n) {
            continue;
        }
        let window = (n + 5).min(2 * n);
        let mut start = i;
        while start > 0 && i - start < window && atoms[start - 1].is_term() {
            start -= 1;
        }
        if start == i {
            continue;
        }
        if let Some(offset) = best_long_form(letters, &atoms[start..i])
            && (i - (start + offset) >= 2 || n <= 3) {
                defs.push(Definition { short: i + 1, long: start + offset..i });
            }
    }
    defs
}

/// Schwartz–Hearst: match short-form characters right-to-left against the
/// candidate words; the first character must start a word. Returns the index
/// of the first long-form atom within `words`.
fn best_long_form(short: &str, words: &[Atom]) -> Option<usize> {
    // flatten to (char, atom index, starts word)
    let mut chars: Vec<(char, usize, bool)> = Vec::new();
    for (w, atom) in words.iter().enumerate() {
        for (j, c) in atom.text.chars().enumerate() {
            chars.extend(c.to_lowercase().map(|l| (l, w, j == 0)));
        }
    }
    let short: Vec<char> = short.chars().flat_map(char::to_lowercase).collect();
    let mut l = chars.len() as isize - 1;
    for (s, &sc) in short.iter().enumerate().rev() {
        if !sc.is_alphanumeric() {
            continue;
        }
        while l >= 0 && (chars[l as usize].0 != sc || (s == 0 && !chars[l as usize].2)) {
            l -= 1;
        }
        if l < 0 {
            return None;
        }
        l -= 1;
    }
    Some(chars[(l + 1) as usize].1)
}

/// Insert `[BREAK, long form…, BREAK]` before every later occurrence of each
/// defined acronym, unless `known` says the lexicon already handles it.
pub fn expand<'a>(atoms: Vec<Atom<'a>>, known: impl Fn(&Atom) -> bool) -> Vec<Atom<'a>> {
    let defs = definitions(&atoms);
    if defs.is_empty() {
        return atoms;
    }
    let mut out = Vec::with_capacity(atoms.len() + defs.len() * 8);
    let mut active: Vec<&Definition> = Vec::new();
    let mut next_def = defs.iter().peekable();
    for (i, atom) in atoms.iter().enumerate() {
        if next_def.peek().is_some_and(|d| d.short < i) {
            let d = next_def.next().unwrap();
            let short = singularize(atoms[d.short].text);
            if !known(&atoms[d.short]) && !active.iter().any(|a| singularize(atoms[a.short].text) == short) {
                active.push(d);
            }
        }
        if atom.kind == Kind::Acronym {
            let short = singularize(atom.text);
            // inside a compound (`ADAPT-VQE`) an inserted break would split it
            let in_compound = out.last().is_some_and(|a: &Atom| a.hyphen_next);
            if let Some(d) = active.iter().find(|d| singularize(atoms[d.short].text) == short).filter(|_| !in_compound) {
                out.push(BREAK);
                out.extend_from_slice(&atoms[d.long.clone()]);
                if let Some(last) = out.last_mut() {
                    last.hyphen_next = false;
                }
                out.push(BREAK);
            }
        }
        out.push(*atom);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn long_forms(text: &str) -> Vec<(String, String)> {
        let atoms = lex(text);
        definitions(&atoms)
            .into_iter()
            .map(|d| {
                let long: Vec<&str> = atoms[d.long].iter().map(|a| a.text).collect();
                (atoms[d.short].text.to_owned(), long.join(" "))
            })
            .collect()
    }

    #[test]
    fn detects_definitions() {
        assert_eq!(
            long_forms("We study the variational quantum eigensolver (VQE) on hardware."),
            vec![("VQE".into(), "variational quantum eigensolver".into())]
        );
        assert_eq!(
            long_forms("the quantum approximate optimization algorithm (QAOA)"),
            vec![("QAOA".into(), "quantum approximate optimization algorithm".into())]
        );
        assert_eq!(
            long_forms("noisy intermediate-scale quantum (NISQ) devices"),
            vec![("NISQ".into(), "noisy intermediate scale quantum".into())]
        );
        assert_eq!(long_forms("quantum processing units (QPUs)"), vec![("QPUs".into(), "quantum processing units".into())]);
        assert!(long_forms("we find (QAOA) works").is_empty());
        assert!(long_forms("a random sentence (XYZ)").is_empty());
    }

    #[test]
    fn expands_later_uses_only() {
        let atoms = lex("variational quantum eigensolver (VQE). VQE converges");
        let texts: Vec<&str> = expand(atoms, |_| false).iter().map(|a| a.text).collect();
        assert_eq!(
            texts,
            vec!["variational", "quantum", "eigensolver", "", "VQE", "", "", "", "variational", "quantum", "eigensolver", "", "VQE", "converges"]
        );
    }

    #[test]
    fn compounds_are_not_split() {
        let atoms = lex("variational quantum eigensolver (VQE). ADAPT-VQE");
        let out = expand(atoms, |_| false);
        assert_eq!(out[out.len() - 2].text, "ADAPT");
        assert!(out[out.len() - 2].hyphen_next);
    }

    #[test]
    fn known_acronyms_are_not_expanded() {
        let atoms = lex("variational quantum eigensolver (VQE). VQE");
        assert_eq!(expand(atoms.clone(), |_| true), atoms);
    }
}
