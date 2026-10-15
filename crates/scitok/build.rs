//! Generates the registry of built-in domains from `data/domains/*`. Each
//! domain is compiled in only when its `domain-<name>` cargo feature is on.

use std::fmt::Write;
use std::path::Path;
use std::{env, fs};

const FILES: [&str; 4] = ["curated.toml", "mined.toml", "stats.toml", "stopwords.txt"];

fn main() {
    let root = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("data/domains");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut dirs: Vec<_> = fs::read_dir(&root)
        .map(|rd| rd.filter_map(Result::ok).filter(|e| e.path().is_dir()).collect())
        .unwrap_or_default();
    dirs.sort_by_key(|e| e.file_name());

    let mut code = String::from("pub(crate) static BUILTIN: &[(&str, [Option<&str>; 4])] = &[\n");
    for dir in dirs {
        let name = dir.file_name().into_string().unwrap();
        let feature = format!("CARGO_FEATURE_DOMAIN_{}", name.to_uppercase().replace(['-', '.'], "_"));
        if env::var_os(&feature).is_none() {
            continue;
        }
        write!(code, "    ({name:?}, [").unwrap();
        for file in FILES {
            let path = dir.path().join(file);
            println!("cargo:rerun-if-changed={}", path.display());
            if path.exists() {
                write!(code, "Some(include_str!({:?})), ", path.display().to_string()).unwrap();
            } else {
                code.push_str("None, ");
            }
        }
        code.push_str("]),\n");
    }
    code.push_str("];\n");
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("builtin_domains.rs"), code).unwrap();
}
