//! Hand-written scanner turning normalized text into atoms.
//!
//! Hyphenated compounds are split into their parts, with `hyphen_next` set on
//! every part but the last, so `Lieb-Robinson` and `Lieb Robinson` produce the
//! same atom sequence for lexicon matching.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Plain word (at most one uppercase letter): `qubit`, `Grover`.
    Word,
    /// Two or more uppercase letters: `QAOA`, `qLDPC`, `CNOTs`, `MaxCut`.
    Acronym,
    /// Letters and digits: `T1`, `2D`, `Z_2`.
    AlphaNum,
    /// Digits only.
    Number,
    /// Lie-group-like symbol: `SU(2)`, `U(1)`, `SO(N)`.
    Group,
    /// Punctuation that ends a phrase.
    Break,
    Open,
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atom<'a> {
    pub text: &'a str,
    pub kind: Kind,
    pub hyphen_next: bool,
}

impl Atom<'_> {
    pub fn is_term(&self) -> bool {
        !matches!(self.kind, Kind::Break | Kind::Open | Kind::Close)
    }

    const fn punct(kind: Kind) -> Atom<'static> {
        Atom { text: "", kind, hyphen_next: false }
    }
}

pub const BREAK: Atom<'static> = Atom::punct(Kind::Break);

pub fn lex(text: &str) -> Vec<Atom<'_>> {
    let bytes = text.as_bytes();
    let mut atoms: Vec<Atom> = Vec::with_capacity(text.len() / 6);
    let mut i = 0;
    while i < text.len() {
        let c = char_at(text, i);
        if c.is_alphanumeric() {
            let start = i;
            i += c.len_utf8();
            while i < text.len() {
                let d = char_at(text, i);
                if d.is_alphanumeric() {
                    i += d.len_utf8();
                } else if d == '_' && next_is_alnum(text, i + 1) {
                    i += 1;
                } else {
                    break;
                }
            }
            let mut kind = classify(&text[start..i]);
            if let Some(end) = group_suffix(text, start, i) {
                i = end;
                kind = Kind::Group;
            }
            let word_end = i;
            // Possessives: Grover's, qubits'
            if bytes.get(i) == Some(&b'\'') {
                if bytes.get(i + 1) == Some(&b's') && !next_is_alnum(text, i + 2) {
                    i += 2;
                } else if !next_is_alnum(text, i + 1) {
                    i += 1;
                }
            }
            // `-`, or LaTeX's `--` en dash as in `Lieb--Robinson`
            let dashes = match (bytes.get(i), bytes.get(i + 1)) {
                (Some(b'-'), Some(b'-')) => 2,
                (Some(b'-'), _) => 1,
                _ => 0,
            };
            let hyphen_next = dashes > 0 && next_is_alnum(text, i + dashes);
            if hyphen_next {
                i += dashes;
            }
            atoms.push(Atom { text: &text[start..word_end], kind, hyphen_next });
            continue;
        }
        i += c.len_utf8();
        let atom = match c {
            '(' | '[' => Atom::punct(Kind::Open),
            ')' | ']' => Atom::punct(Kind::Close),
            '\'' => continue,
            '-' if next_is_alnum(text, i) => continue,
            c if c.is_whitespace() => continue,
            _ => BREAK,
        };
        if atom.kind == Kind::Break && atoms.last().is_some_and(|a| a.kind == Kind::Break) {
            continue;
        }
        atoms.push(atom);
    }
    atoms
}

fn char_at(text: &str, i: usize) -> char {
    let b = text.as_bytes()[i];
    if b.is_ascii() {
        b as char
    } else {
        text[i..].chars().next().unwrap()
    }
}

fn next_is_alnum(text: &str, i: usize) -> bool {
    i < text.len() && text.is_char_boundary(i) && char_at(text, i).is_alphanumeric()
}

fn classify(word: &str) -> Kind {
    let (mut digits, mut upper, mut letters) = (0, 0, 0);
    for c in word.chars() {
        if c.is_numeric() {
            digits += 1;
        } else if c.is_alphabetic() {
            letters += 1;
            if c.is_uppercase() {
                upper += 1;
            }
        }
    }
    if letters == 0 && digits > 0 {
        Kind::Number
    } else if digits > 0 {
        Kind::AlphaNum
    } else if upper >= 2 {
        Kind::Acronym
    } else {
        Kind::Word
    }
}

/// `SU` followed by `(2)` / `(N)`: returns the end of the group symbol.
fn group_suffix(text: &str, start: usize, end: usize) -> Option<usize> {
    let name = &text.as_bytes()[start..end];
    if name.is_empty() || name.len() > 3 || name == b"O" || !name.iter().all(u8::is_ascii_uppercase) {
        return None;
    }
    let rest = &text.as_bytes()[end..];
    if rest.first() != Some(&b'(') {
        return None;
    }
    let close = rest.iter().take(5).position(|&b| b == b')')?;
    let arg = &rest[1..close];
    let ok = !arg.is_empty()
        && (arg.iter().all(u8::is_ascii_digit) || (arg.len() == 1 && arg[0].is_ascii_alphabetic()));
    ok.then_some(end + close + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<(&str, Kind, bool)> {
        lex(s).into_iter().map(|a| (a.text, a.kind, a.hyphen_next)).collect()
    }

    #[test]
    fn splits_hyphenated_compounds() {
        assert_eq!(
            texts("Lieb-Robinson bound"),
            vec![("Lieb", Kind::Word, true), ("Robinson", Kind::Word, false), ("bound", Kind::Word, false)]
        );
    }

    #[test]
    fn latex_en_dash() {
        assert_eq!(texts("Lieb--Robinson"), texts("Lieb-Robinson"));
        assert!(!lex("a---b")[0].hyphen_next);
    }

    #[test]
    fn strips_possessives() {
        assert_eq!(texts("Grover's algorithm")[0], ("Grover", Kind::Word, false));
        assert_eq!(texts("qubits' states")[0], ("qubits", Kind::Word, false));
        assert_eq!(texts("Shor's-type")[0], ("Shor", Kind::Word, true));
    }

    #[test]
    fn classifies_kinds() {
        let kinds: Vec<Kind> = lex("QAOA qLDPC T1 2D 42 SU(2) U(N) Z_2 qubit").iter().map(|a| a.kind).collect();
        use Kind::*;
        assert_eq!(kinds, vec![Acronym, Acronym, AlphaNum, AlphaNum, Number, Group, Group, AlphaNum, Word]);
        assert_eq!(lex("SU(2) symmetry")[0].text, "SU(2)");
    }

    #[test]
    fn punctuation() {
        use Kind::*;
        let kinds: Vec<Kind> = lex("eigensolver (VQE), which; -qubit").iter().map(|a| a.kind).collect();
        assert_eq!(kinds, vec![Word, Open, Acronym, Close, Break, Word, Break, Word]);
    }
}
