//! On-disk harvest cache: `<cache>/<category>/abstracts.jsonl`, the OAI-PMH
//! resumption state, and full texts under `fulltext/`.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Record {
    pub id: String,
    pub datestamp: String,
    #[serde(default)]
    pub created: Option<String>,
    pub categories: Vec<String>,
    pub title: String,
    #[serde(rename = "abstract")]
    pub abstract_: String,
}

/// Where an interrupted harvest left off.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct HarvestState {
    pub set: String,
    pub from: String,
    pub until: String,
    pub token: Option<String>,
    pub done: bool,
}

pub struct Cache {
    dir: PathBuf,
    ids: HashSet<String>,
}

impl Cache {
    pub fn open(root: &Path, category: &str) -> Result<Cache, String> {
        let dir = root.join(category);
        fs::create_dir_all(dir.join("fulltext")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut cache = Cache { dir, ids: HashSet::new() };
        cache.ids = cache.records()?.into_iter().map(|r| r.id).collect();
        Ok(cache)
    }

    fn abstracts_path(&self) -> PathBuf {
        self.dir.join("abstracts.jsonl")
    }

    fn state_path(&self) -> PathBuf {
        self.dir.join("harvest.json")
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Append records not cached yet; returns how many were new.
    pub fn append(&mut self, records: &[Record]) -> Result<usize, String> {
        let path = self.abstracts_path();
        let torn = last_byte(&path).is_some_and(|c| c != b'\n');
        let mut file = OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if torn {
            writeln!(file).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        let mut added = 0;
        for r in records {
            if self.ids.insert(r.id.clone()) {
                let line = serde_json::to_string(r).unwrap();
                writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
                added += 1;
            }
        }
        Ok(added)
    }

    /// All cached records, first occurrence of each id.
    pub fn records(&self) -> Result<Vec<Record>, String> {
        let path = self.abstracts_path();
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for (n, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            // a crash mid-write can leave a truncated last line
            let Ok(r) = serde_json::from_str::<Record>(&line) else {
                eprintln!("warning: skipping malformed line {} of {}", n + 1, path.display());
                continue;
            };
            if seen.insert(r.id.clone()) {
                out.push(r);
            }
        }
        Ok(out)
    }

    pub fn state(&self) -> Option<HarvestState> {
        let text = fs::read_to_string(self.state_path()).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save_state(&self, state: &HarvestState) -> Result<(), String> {
        let path = self.state_path();
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_string_pretty(state).unwrap()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// `fulltext/<id>.<ext>`; not `Path::with_extension`, since new-style ids
    /// (`2610.00669`) contain a dot.
    fn fulltext_path(&self, id: &str, ext: &str) -> PathBuf {
        self.dir.join("fulltext").join(format!("{}.{ext}", id.replace('/', "_")))
    }

    /// Cached LaTeX source, if any.
    pub fn fulltext(&self, id: &str) -> Option<String> {
        fs::read_to_string(self.fulltext_path(id, "tex")).ok()
    }

    /// Has a fetch been attempted (LaTeX stored, or marked as unavailable)?
    pub fn fulltext_attempted(&self, id: &str) -> bool {
        self.fulltext_path(id, "tex").exists() || self.fulltext_path(id, "skip").exists()
    }

    pub fn store_fulltext(&self, id: &str, latex: Option<&str>, reason: &str) -> Result<(), String> {
        let (path, content) = match latex {
            Some(tex) => (self.fulltext_path(id, "tex"), tex),
            None => (self.fulltext_path(id, "skip"), reason),
        };
        fs::write(&path, content).map_err(|e| format!("{}: {e}", path.display()))
    }
}

fn last_byte(path: &Path) -> Option<u8> {
    let mut f = File::open(path).ok()?;
    f.seek(SeekFrom::End(-1)).ok()?;
    let mut b = [0u8];
    f.read_exact(&mut b).ok()?;
    Some(b[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("scitok-learn-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn appends_and_dedupes() {
        let root = temp_dir("cache");
        let rec = |id: &str| Record { id: id.into(), title: format!("t{id}"), ..Record::default() };
        let mut c = Cache::open(&root, "quant-ph").unwrap();
        assert_eq!(c.append(&[rec("1"), rec("2"), rec("1")]).unwrap(), 2);
        // a torn write from a crash
        fs::OpenOptions::new().append(true).open(c.abstracts_path()).unwrap().write_all(b"{\"id\": \"3\", \"tit").unwrap();
        let mut c = Cache::open(&root, "quant-ph").unwrap();
        assert_eq!(c.len(), 2);
        assert!(c.ids.contains("2"));
        assert_eq!(c.append(&[rec("4")]).unwrap(), 1);
        assert_eq!(Cache::open(&root, "quant-ph").unwrap().len(), 3);
        let state = HarvestState { set: "s".into(), token: Some("tok".into()), ..HarvestState::default() };
        c.save_state(&state).unwrap();
        assert_eq!(c.state(), Some(state));
        c.store_fulltext("quant-ph/0101001", None, "pdf only").unwrap();
        assert!(c.fulltext_attempted("quant-ph/0101001"));
        assert_eq!(c.fulltext("quant-ph/0101001"), None);
        c.store_fulltext("2610.00669", Some("a"), "").unwrap();
        c.store_fulltext("2610.02090", Some("b"), "").unwrap();
        assert_eq!(c.fulltext("2610.00669").as_deref(), Some("a"));
        assert_eq!(c.fulltext("2610.02090").as_deref(), Some("b"));
        assert!(!c.fulltext_attempted("2610.06984"));
        fs::remove_dir_all(root).unwrap();
    }
}
