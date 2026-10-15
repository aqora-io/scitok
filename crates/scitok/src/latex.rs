//! LaTeX → plain text. A linear scanner, not a parser: it keeps prose and the
//! text arguments of formatting commands, drops references, citations, floats
//! and math, and keeps only a few meaningful symbols from math (`SU(2)`,
//! `\mathbb{Z}_2`, single-letter names as in `$T$-gate`).

use crate::normalize::greek_name;

/// Commands whose `[..]`/`{..}` arguments are dropped together with the command.
const DROP_ARGS: &[&str] = &[
    "cite", "citep", "citet", "citealp", "citeauthor", "citeyear", "nocite", "ref", "eqref",
    "autoref", "cref", "Cref", "pageref", "label", "url", "includegraphics", "bibliography",
    "bibliographystyle", "usepackage", "documentclass", "newcommand", "renewcommand",
    "providecommand", "newenvironment", "renewenvironment", "newtheorem", "DeclareMathOperator",
    "affiliation", "affil", "author", "email", "address", "date", "thanks", "vspace", "hspace",
    "setlength", "addtolength", "setcounter", "addtocounter", "pacs", "keywords", "input",
    "include", "graphicspath", "hypersetup", "definecolor", "color", "textcolor", "orcid",
    "doi", "arxiv", "linenumbers", "bibitem",
];

/// Environments dropped entirely.
const DROP_ENVS: &[&str] = &[
    "thebibliography", "verbatim", "lstlisting", "tikzpicture", "comment", "tabular",
    "tabular*", "tabularx", "minted", "algorithmic", "quantikz", "circuitikz",
];

/// Environments whose body is math.
const MATH_ENVS: &[&str] = &[
    "equation", "equation*", "align", "align*", "eqnarray", "eqnarray*", "gather", "gather*",
    "multline", "multline*", "displaymath", "math", "flalign", "flalign*", "alignat",
    "alignat*", "split", "subequations",
];

/// Floats: only `\caption` text survives.
const FLOAT_ENVS: &[&str] = &["figure", "figure*", "table", "table*", "wrapfigure", "algorithm"];

/// Convert LaTeX to plain text. With `full_document`, the preamble and anything
/// after `\end{document}` are dropped (apart from `\title`).
pub fn to_text(src: &str, full_document: bool) -> String {
    let mut out = String::with_capacity(src.len());
    let body = if full_document {
        match src.find("\\begin{document}") {
            Some(start) => {
                if let Some(title) = command_arg(&src[..start], "title") {
                    Scanner::new(title).run(&mut out);
                    out.push_str(".\n");
                }
                let rest = &src[start + "\\begin{document}".len()..];
                &rest[..rest.find("\\end{document}").unwrap_or(rest.len())]
            }
            None => src,
        }
    } else {
        src
    };
    let mut scanner = Scanner::new(body);
    scanner.comments = full_document;
    scanner.run(&mut out);
    out
}

/// Argument of the first `\name{..}` in `src`.
fn command_arg<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("\\{name}");
    let mut from = 0;
    while let Some(pos) = src[from..].find(&pat) {
        let after = from + pos + pat.len();
        let rest = &src.as_bytes()[after..];
        if rest.first().is_some_and(u8::is_ascii_alphabetic) {
            from = after;
            continue;
        }
        let mut s = Scanner::new(src);
        s.pos = after;
        s.skip_optional_args();
        return s.brace_group();
    }
    None
}

struct Scanner<'a> {
    src: &'a str,
    pos: usize,
    /// `%` starts a comment. Off for fragments such as arXiv abstracts, where
    /// a bare `%` is a percent sign (`99.9% for $n=2$`).
    comments: bool,
}

impl<'a> Scanner<'a> {
    fn new(src: &'a str) -> Self {
        Scanner { src, pos: 0, comments: true }
    }

    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    /// Scanner for a nested argument, with the same comment handling.
    fn sub(&self, src: &'a str) -> Scanner<'a> {
        Scanner { src, pos: 0, comments: self.comments }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn run(&mut self, out: &mut String) {
        while let Some(b) = self.peek() {
            match b {
                b'%' if self.comments => self.skip_line(),
                b'$' => {
                    let display = self.rest().starts_with("$$");
                    let delim = if display { "$$" } else { "$" };
                    self.pos += delim.len();
                    let math = self.take_until(delim);
                    push_math(math, out);
                }
                b'\\' => self.command(out),
                b'{' | b'}' => self.pos += 1,
                b'~' | b'&' => {
                    out.push(' ');
                    self.pos += 1;
                }
                _ => {
                    let start = self.pos;
                    // the current char may be a literal `%`: search after it
                    let first = self.rest().chars().next().map_or(1, char::len_utf8);
                    let end = self.rest()[first..]
                        .find(['%', '$', '\\', '{', '}', '~', '&'])
                        .map_or(self.src.len(), |i| start + first + i);
                    out.push_str(&self.src[start..end]);
                    self.pos = end;
                }
            }
        }
    }

    fn skip_line(&mut self) {
        self.pos = self.rest().find('\n').map_or(self.src.len(), |i| self.pos + i + 1);
    }

    /// Text up to `delim` (unescaped); the cursor moves past the delimiter.
    fn take_until(&mut self, delim: &str) -> &'a str {
        let start = self.pos;
        let mut i = start;
        let bytes = self.src.as_bytes();
        while i < bytes.len() {
            if bytes[i..].starts_with(delim.as_bytes()) {
                self.pos = i + delim.len();
                return &self.src[start..i];
            }
            i += if bytes[i] == b'\\' { 2 } else { 1 };
        }
        self.pos = self.src.len();
        &self.src[start.min(self.src.len())..]
    }

    fn command(&mut self, out: &mut String) {
        self.pos += 1;
        let Some(c) = self.rest().chars().next() else { return };
        if !c.is_ascii_alphabetic() {
            self.pos += c.len_utf8();
            match c {
                '(' => push_math(self.take_until("\\)"), out),
                '[' => push_math(self.take_until("\\]"), out),
                '%' | '$' | '#' | '_' | '{' | '}' => out.push(c),
                '&' => out.push_str(" and "),
                '"' | '\'' | '`' | '^' | '~' | '=' | '.' => self.accent(out),
                '-' => {} // discretionary hyphen
                _ => out.push(' '),
            }
            return;
        }
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_alphabetic()) {
            self.pos += 1;
        }
        let name = &self.src[start..self.pos];
        if self.peek() == Some(b'*') {
            self.pos += 1;
        }
        match name {
            "begin" => self.begin(out),
            "end" => {
                self.brace_group();
            }
            "item" | "par" | "newline" | "linebreak" | "maketitle" | "noindent" => {
                self.skip_optional_args();
                out.push('\n');
            }
            "ss" => out.push_str("ss"),
            "aa" | "AA" => out.push_str(&name[..1]),
            "o" | "O" | "i" | "l" | "L" | "ae" | "AE" | "oe" | "OE" => {
                out.push_str(&name[..1]);
                if name.len() == 2 {
                    out.push_str(&name[1..]);
                }
            }
            "v" | "u" | "H" | "c" | "k" | "r" | "d" | "b" | "t" if self.at_accent_target() => {
                self.accent(out)
            }
            "href" => {
                self.brace_group();
                if let Some(text) = self.brace_group() {
                    self.sub(text).run(out);
                }
            }
            "footnote" | "caption" | "section" | "subsection" | "subsubsection" | "paragraph"
            | "chapter" | "title" => {
                // keep the text, but end the sentence around it
                self.skip_optional_args();
                out.push_str(".\n");
                if let Some(text) = self.brace_group() {
                    self.sub(text).run(out);
                }
                out.push_str(".\n");
            }
            _ if DROP_ARGS.contains(&name) => self.skip_args(),
            _ => match greek_macro(name) {
                Some(g) => {
                    out.push(' ');
                    out.push_str(g);
                    out.push(' ');
                }
                // unknown/formatting command: drop the name, keep argument text
                None => {
                    if self.peek() == Some(b' ') {
                        self.pos += 1;
                    }
                }
            },
        }
    }

    fn at_accent_target(&self) -> bool {
        let rest = self.rest().as_bytes();
        match rest.first() {
            Some(b'{') => true,
            Some(b' ') => rest.get(1).is_some_and(u8::is_ascii_alphabetic) && !rest.get(2).is_some_and(u8::is_ascii_alphabetic),
            _ => false,
        }
    }

    /// After an accent command: emit the accented letter without its accent.
    fn accent(&mut self, out: &mut String) {
        while self.peek() == Some(b' ') {
            self.pos += 1;
        }
        if self.peek() == Some(b'{') {
            if let Some(arg) = self.brace_group() {
                out.push_str(arg.trim_start_matches('\\'));
            }
        } else if let Some(c) = self.rest().chars().next() {
            if c == '\\' {
                // \"\i
                self.pos += 1;
            } else {
                out.push(c);
                self.pos += c.len_utf8();
            }
        }
    }

    fn begin(&mut self, out: &mut String) {
        let Some(env) = self.brace_group() else { return };
        let end_tag = format!("\\end{{{env}}}");
        if DROP_ENVS.contains(&env) {
            self.take_until(&end_tag);
        } else if MATH_ENVS.contains(&env) {
            let body = self.take_until(&end_tag);
            push_math(body, out);
        } else if FLOAT_ENVS.contains(&env) {
            let body = self.take_until(&end_tag);
            let mut s = self.sub(body);
            while let Some(pos) = s.rest().find("\\caption") {
                s.pos += pos + "\\caption".len();
                s.skip_optional_args();
                if let Some(caption) = s.brace_group() {
                    out.push_str(".\n");
                    self.sub(caption).run(out);
                    out.push_str(".\n");
                }
            }
        } else {
            // abstract, itemize, theorem, ...: keep the body, drop `[title]`
            self.skip_optional_args();
            out.push('\n');
        }
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|b| b == b' ' || b == b'\t' || b == b'\n') {
            self.pos += 1;
        }
    }

    fn skip_optional_args(&mut self) {
        let save = self.pos;
        self.skip_ws();
        if self.peek() == Some(b'[') {
            self.balanced(b'[', b']');
        } else {
            self.pos = save;
        }
    }

    /// Skip all following `[..]`/`{..}` groups, and a `\macro` name as used in
    /// `\newcommand\foo{..}`.
    fn skip_args(&mut self) {
        let mut first = true;
        loop {
            let save = self.pos;
            self.skip_ws();
            match self.peek() {
                Some(b'[') => {
                    self.balanced(b'[', b']');
                }
                Some(b'{') => {
                    self.balanced(b'{', b'}');
                }
                Some(b'\\') if first && self.rest().as_bytes().get(1).is_some_and(u8::is_ascii_alphabetic) => {
                    self.pos += 1;
                    while self.peek().is_some_and(|b| b.is_ascii_alphabetic()) {
                        self.pos += 1;
                    }
                }
                _ => {
                    self.pos = save;
                    return;
                }
            }
            first = false;
        }
    }

    /// `{..}` at the cursor (after optional whitespace): returns the content.
    fn brace_group(&mut self) -> Option<&'a str> {
        let save = self.pos;
        self.skip_ws();
        if self.peek() != Some(b'{') {
            self.pos = save;
            return None;
        }
        self.balanced(b'{', b'}')
    }

    fn balanced(&mut self, open: u8, close: u8) -> Option<&'a str> {
        let bytes = self.src.as_bytes();
        let start = self.pos + 1;
        let mut depth = 0usize;
        let mut i = self.pos;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 1,
                b if b == open => depth += 1,
                b if b == close => {
                    depth -= 1;
                    if depth == 0 {
                        self.pos = i + 1;
                        return Some(&self.src[start..i]);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        self.pos = bytes.len();
        Some(&self.src[start.min(bytes.len())..])
    }
}

fn greek_macro(name: &str) -> Option<&'static str> {
    let base = name.strip_prefix("var").unwrap_or(name);
    let lower = base.to_ascii_lowercase();
    ('α'..='ω').filter_map(greek_name).find(|g| *g == lower)
}

/// Keep only what carries meaning in math: `\text{..}`, group symbols such as
/// `SU(2)` or `\mathbb{Z}_2`, and a lone variable like the `T` in `$T$-gate`.
fn push_math(math: &str, out: &mut String) {
    let trimmed = math.trim();
    if trimmed.len() == 1 && trimmed.as_bytes()[0].is_ascii_alphabetic() {
        out.push_str(trimmed);
        return;
    }
    out.push(' ');
    let flat = flatten_math(math, out);
    let b = flat.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_uppercase() && (i == 0 || !b[i - 1].is_ascii_alphabetic()) {
            let start = i;
            while i < b.len() && b[i].is_ascii_uppercase() && i - start < 3 {
                i += 1;
            }
            if i < b.len() && !b[i].is_ascii_alphabetic() {
                if b[i] == b'(' && &flat[start..i] != "O" {
                    if let Some(len) = flat[i..].find(')') {
                        let arg = &flat[i + 1..i + len];
                        let arg_ok = arg.bytes().all(|c| c.is_ascii_digit()) && !arg.is_empty()
                            || arg.len() == 1 && arg.as_bytes()[0].is_ascii_alphabetic();
                        if arg_ok {
                            out.push_str(&flat[start..=i + len]);
                            out.push(' ');
                            i += len + 1;
                            continue;
                        }
                    }
                } else if b[i] == b'_' && i - start == 1 && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
                    // Z_2, Z_N-like discrete groups (single digit subscripts)
                    out.push_str(&flat[start..i + 2]);
                    out.push(' ');
                    i += 2;
                    continue;
                }
            }
            continue;
        }
        i += 1;
    }
}

/// Strip font wrappers (`\mathrm{SU}` → `SU`), braces and spaces; `\text{..}`
/// content goes straight to `out`.
fn flatten_math(math: &str, out: &mut String) -> String {
    let mut flat = String::with_capacity(math.len());
    let mut s = Scanner::new(math);
    while let Some(b) = s.peek() {
        match b {
            b'\\' => {
                s.pos += 1;
                let start = s.pos;
                while s.peek().is_some_and(|b| b.is_ascii_alphabetic()) {
                    s.pos += 1;
                }
                let name = &math[start..s.pos];
                if name.is_empty() {
                    s.pos += s.rest().chars().next().map_or(0, char::len_utf8);
                    flat.push(' ');
                } else if matches!(name, "text" | "textrm" | "textit" | "textbf" | "mbox" | "textsc") {
                    if let Some(t) = s.brace_group() {
                        out.push(' ');
                        Scanner::new(t).run(out);
                        out.push(' ');
                    }
                } else if !matches!(name, "mathrm" | "mathbb" | "mathcal" | "mathsf" | "mathbf" | "rm" | "operatorname" | "mathfrak" | "mathit") {
                    flat.push(' ');
                }
            }
            b'{' | b'}' | b' ' => s.pos += 1,
            _ => {
                let c = s.rest().chars().next().unwrap();
                flat.push(c);
                s.pos += c.len_utf8();
            }
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squash(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn drops_citations_and_refs() {
        assert_eq!(
            squash(&to_text("Grover's algorithm~\\cite{grover96} gives $O(\\sqrt{N})$ queries (see Fig.~\\ref{fig:a}).", false)),
            "Grover's algorithm gives queries (see Fig. )."
        );
    }

    #[test]
    fn accents() {
        assert_eq!(to_text("Schr\\\"odinger and Schr\\\"{o}dinger, Erd\\H{o}s, \\v{C}ech", false), "Schrodinger and Schrodinger, Erdos, Cech");
    }

    #[test]
    fn keeps_formatting_arguments() {
        assert_eq!(squash(&to_text("a \\emph{surface code} and \\textbf{toric} code", false)), "a surface code and toric code");
        assert_eq!(squash(&to_text("\\section{Introduction} Text", false)), ". Introduction. Text");
    }

    #[test]
    fn math_symbols() {
        assert_eq!(squash(&to_text("the $T$-gate and $\\mathrm{SU}(2)$ and $\\mathbb{Z}_2$ order", false)), "the T-gate and SU(2) and Z_2 order");
        assert_eq!(squash(&to_text("$E = \\langle H \\rangle \\text{ for all } t$", false)), "for all");
        assert_eq!(squash(&to_text("\\begin{equation} U(1) \\label{eq} \\end{equation} after", false)), "U(1) after");
    }

    #[test]
    fn comments_floats_and_preamble() {
        let doc = "\\documentclass{revtex4}\n\\usepackage{amsmath}\n\\title{Fast \\emph{QAOA}}\n\\begin{document}\n% secret\nBody text.\n\\begin{figure}\\includegraphics{x.pdf}\\caption[short]{A \\textit{surface code} lattice.}\\label{f}\\end{figure}\n\\begin{thebibliography}{9}\\bibitem{a} Junk\\end{thebibliography}\n\\end{document}\ntrailing";
        assert_eq!(squash(&to_text(doc, true)), "Fast QAOA. Body text. . A surface code lattice..");
    }

    #[test]
    fn newcommand_is_dropped() {
        assert_eq!(squash(&to_text("\\newcommand{\\ket}[1]{|#1\\rangle} \\newcommand\\bra[1]{x} text", false)), "text");
    }

    #[test]
    fn never_panics_on_odd_input() {
        let inputs = [
            "\\γ", "$γ$", "\\(γ\\)", "\\[γ", "{γ", "\\emph{γ", "$$", "$", "\\", "\\begin{figure}γ",
            "\\caption", "\\\"", "\\\"{", "\\v", "%γ", "\\href{γ}", "\\newcommand", "$\\γ\\text{β}$",
            "ε$SU(γ)$", "$\\mathbb{Z}_$", "$Z_γ$", "$U(", "\\section*{Ω}", "\\begin{", "\\end",
        ];
        for input in inputs {
            to_text(input, false);
            to_text(input, true);
        }
    }

    #[test]
    fn percent_in_fragments() {
        assert_eq!(to_text("reaches 99.9% for $n=2$ qubits", false).split_whitespace().collect::<Vec<_>>(), ["reaches", "99.9%", "for", "qubits"]);
        assert_eq!(to_text("H{\\aa}kan {\\AA}ngstrom", false), "Hakan Angstrom");
    }

    #[test]
    fn greek_macros_in_text() {
        assert_eq!(squash(&to_text("the \\alpha-particle", false)), "the alpha -particle");
    }
}
