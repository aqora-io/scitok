//! Optional full-text download: arXiv e-print sources (gzipped tarball or a
//! single gzipped `.tex`), reduced to one LaTeX document with `\input`s inlined.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::GzDecoder;

use crate::cache::{Cache, Record};
use crate::http::{Client, Clock, Transport};

pub const DEFAULT_SOURCE_URL: &str = "https://export.arxiv.org/src";

/// Decompressed size limit for one e-print.
const MAX_UNPACKED: u64 = 256 * 1024 * 1024;

#[derive(Debug, Default)]
pub struct FulltextStats {
    pub fetched: usize,
    pub skipped: usize,
    pub cached: usize,
}

pub fn fetch_all<T: Transport, C: Clock>(
    client: &mut Client<T, C>,
    cache: &Cache,
    base_url: &str,
    records: &[Record],
    max: Option<usize>,
) -> Result<FulltextStats, String> {
    let mut stats = FulltextStats::default();
    let todo: Vec<&Record> = records.iter().filter(|r| !cache.fulltext_attempted(&r.id)).collect();
    stats.cached = records.len() - todo.len();
    eprintln!("full text: {} cached, {} to fetch", stats.cached, todo.len());
    for r in todo.into_iter().take(max.unwrap_or(usize::MAX)) {
        let url = format!("{base_url}/{}", r.id);
        let resp = client.get(&url)?;
        let pdf = resp.content_type.as_deref().is_some_and(|t| t.contains("pdf"));
        let (latex, reason) = if resp.status == 404 || resp.status == 410 {
            (None, format!("HTTP {}", resp.status))
        } else if resp.status != 200 {
            // e.g. 403 when blocked: don't mark the paper as done
            return Err(format!("GET {url}: HTTP {}; stopping, rerun later to continue", resp.status));
        } else if pdf {
            (None, "PDF-only submission".to_owned())
        } else {
            match extract_latex(&resp.body) {
                Ok(Some(tex)) => (Some(tex), String::new()),
                Ok(None) => (None, "no LaTeX source (PDF-only or unrecognized)".to_owned()),
                Err(e) => (None, e),
            }
        };
        if latex.is_some() {
            stats.fetched += 1;
        } else {
            eprintln!("  {}: {reason}", r.id);
            stats.skipped += 1;
        }
        cache.store_fulltext(&r.id, latex.as_deref(), &reason)?;
    }
    Ok(stats)
}

/// LaTeX of an e-print payload, or `None` when it is not LaTeX (e.g. a PDF).
pub fn extract_latex(body: &[u8]) -> Result<Option<String>, String> {
    if body.starts_with(b"%PDF") {
        return Ok(None);
    }
    let data = if body.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        GzDecoder::new(body).take(MAX_UNPACKED).read_to_end(&mut out).map_err(|e| format!("gunzip: {e}"))?;
        out
    } else {
        body.to_vec()
    };
    if data.starts_with(b"%PDF") {
        return Ok(None);
    }
    if data.len() > 262 && &data[257..262] == b"ustar" {
        return from_tar(&data);
    }
    let text = String::from_utf8_lossy(&data);
    Ok(is_main(&text).then(|| text.into_owned()))
}

fn is_main(tex: &str) -> bool {
    tex.contains("\\documentclass") || tex.contains("\\begin{document}")
}

fn from_tar(data: &[u8]) -> Result<Option<String>, String> {
    let mut files: HashMap<String, String> = HashMap::new();
    let mut archive = tar::Archive::new(data);
    for entry in archive.entries().map_err(|e| format!("tar: {e}"))? {
        let mut entry = entry.map_err(|e| format!("tar: {e}"))?;
        let path = entry.path().map_err(|e| format!("tar: {e}"))?.to_string_lossy().into_owned();
        if !path.ends_with(".tex") {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| format!("tar: {e}"))?;
        files.insert(path.trim_start_matches("./").to_owned(), String::from_utf8_lossy(&bytes).into_owned());
    }
    let mut mains: Vec<&String> = files.keys().filter(|k| files[*k].contains("\\documentclass")).collect();
    // prefer a document with a body, then the shortest path (top level)
    mains.sort_by_key(|k| (!files[*k].contains("\\begin{document}"), k.matches('/').count(), k.len(), (*k).clone()));
    let Some(main) = mains.first() else { return Ok(None) };
    Ok(Some(inline_inputs(&files[*main], &files, 0)))
}

/// Replace `\input{x}` / `\include{x}` with the content of `x(.tex)`.
fn inline_inputs(tex: &str, files: &HashMap<String, String>, depth: usize) -> String {
    let mut out = String::with_capacity(tex.len());
    let mut rest = tex;
    while let Some(pos) = rest.find(['\\', '%']) {
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        if rest.starts_with('%') {
            // keep comments as-is (scitok drops them), but don't inline inside them
            let end = rest.find('\n').map_or(rest.len(), |i| i + 1);
            out.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        let cmd = ["\\input", "\\include", "\\subfile"].into_iter().find(|c| {
            rest.starts_with(c) && !rest.as_bytes().get(c.len()).is_some_and(u8::is_ascii_alphabetic)
        });
        let Some(cmd) = cmd else {
            let skip = if rest.len() > 1 { 1 + rest[1..].chars().next().map_or(0, char::len_utf8) } else { 1 };
            out.push_str(&rest[..skip]);
            rest = &rest[skip..];
            continue;
        };
        let after = rest[cmd.len()..].trim_start();
        let name = if let Some(braced) = after.strip_prefix('{') {
            braced.find('}').map(|end| (&braced[..end], &braced[end + 1..]))
        } else {
            // \input file
            let end = after.find(|c: char| c.is_whitespace()).unwrap_or(after.len());
            Some((&after[..end], &after[end..]))
        };
        match name {
            Some((name, remaining)) if depth < 8 => {
                let name = name.trim().trim_start_matches("./");
                let content = files.get(name).or_else(|| files.get(&format!("{name}.tex")));
                if let Some(content) = content {
                    out.push('\n');
                    out.push_str(&inline_inputs(content, files, depth + 1));
                    out.push('\n');
                }
                rest = remaining;
            }
            _ => {
                out.push_str(cmd);
                rest = &rest[cmd.len()..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), Compression::fast());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn tarball(files: &[(&str, &str)]) -> Vec<u8> {
        let mut b = tar::Builder::new(Vec::new());
        for (name, content) in files {
            let mut h = tar::Header::new_ustar();
            h.set_size(content.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            b.append_data(&mut h, name, content.as_bytes()).unwrap();
        }
        b.into_inner().unwrap()
    }

    #[test]
    fn single_gzipped_tex() {
        let tex = "\\documentclass{article}\\begin{document}Hi\\end{document}";
        assert_eq!(extract_latex(&gzip(tex.as_bytes())).unwrap().as_deref(), Some(tex));
    }

    #[test]
    fn pdf_only() {
        assert_eq!(extract_latex(b"%PDF-1.5 ...").unwrap(), None);
        assert_eq!(extract_latex(&gzip(b"%PDF-1.5 ...")).unwrap(), None);
    }

    #[test]
    fn tarball_with_inputs() {
        let tar = tarball(&[
            ("figs/plot.tex", "\\documentclass{standalone}"),
            ("main.tex", "\\documentclass{revtex4}\n\\begin{document}\n\\input{sections/intro}\n% \\input{unused}\n\\include{concl.tex}\n\\end{document}"),
            ("sections/intro.tex", "Intro on \\inputenc-free surface codes."),
            ("concl.tex", "Conclusion."),
        ]);
        let tex = extract_latex(&gzip(&tar)).unwrap().unwrap();
        assert!(tex.contains("Intro on \\inputenc-free surface codes."), "{tex}");
        assert!(tex.contains("Conclusion."), "{tex}");
        assert!(tex.contains("% \\input{unused}"), "{tex}");
        assert!(!tex.contains("standalone"), "{tex}");
    }
}
