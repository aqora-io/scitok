//! Unicode normalization: NFKC (ligatures), diacritic folding, Greek letters to
//! names and unified dashes/apostrophes. Pure-ASCII input takes a fast path.

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

pub fn normalize(text: &str) -> String {
    if text.is_ascii() {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    // Compatibility decomposition per character, then dropping combining
    // marks, equals NFKC followed by diacritic folding.
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            std::iter::once(c).nfkd().for_each(|d| push_folded(d, &mut out));
        }
    }
    out
}

fn push_folded(c: char, out: &mut String) {
    if c.is_ascii() {
        out.push(c);
        return;
    }
    if is_combining_mark(c) {
        return;
    }
    match c {
        // hyphen, non-breaking hyphen, figure dash, en dash, minus
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2212}' => out.push('-'),
        // em dash and horizontal bar separate clauses, not compounds
        '\u{2014}' | '\u{2015}' => out.push(' '),
        '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' => out.push('\''),
        '\u{201C}' | '\u{201D}' | '\u{201E}' => out.push('"'),
        'ß' => out.push_str("ss"),
        'æ' => out.push_str("ae"),
        'Æ' => out.push_str("AE"),
        'ø' => out.push('o'),
        'Ø' => out.push('O'),
        'ł' => out.push('l'),
        'Ł' => out.push('L'),
        _ => match greek_name(c) {
            Some(name) => {
                out.push(' ');
                out.push_str(name);
                out.push(' ');
            }
            None => out.push(c),
        },
    }
}

/// Name of a Greek letter (either case), also used for LaTeX macros.
pub fn greek_name(c: char) -> Option<&'static str> {
    const NAMES: [&str; 25] = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "sigma", "tau", "upsilon",
        "phi", "chi", "psi", "omega",
    ];
    let i = match c {
        'α'..='ω' => c as u32 - 'α' as u32,
        'Α'..='Ω' => c as u32 - 'Α' as u32,
        'ϕ' => return Some("phi"),
        'ϵ' => return Some("epsilon"),
        'ϑ' => return Some("theta"),
        _ => return None,
    };
    // Both ranges have 25 slots with the two sigmas at 17/18 (ς σ, and the
    // unassigned U+03A2 before Σ), so one table serves both cases.
    NAMES.get(i as usize).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_diacritics_and_ligatures() {
        assert_eq!(normalize("Schrödinger"), "Schrodinger");
        assert_eq!(normalize("ﬁdelity"), "fidelity");
        assert_eq!(normalize("Erdős"), "Erdos");
    }

    #[test]
    fn unifies_punctuation() {
        assert_eq!(normalize("Lieb–Robinson"), "Lieb-Robinson");
        assert_eq!(normalize("Grover’s"), "Grover's");
        assert_eq!(normalize("codes—which"), "codes which");
    }

    #[test]
    fn greek_letters() {
        assert_eq!(normalize("α-particle").trim(), "alpha -particle");
        assert_eq!(greek_name('Ω'), Some("omega"));
        assert_eq!(greek_name('ω'), Some("omega"));
        assert_eq!(greek_name('Δ'), Some("delta"));
        assert_eq!(greek_name('ς'), Some("sigma"));
    }
}
