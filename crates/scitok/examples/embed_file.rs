//! Embed every line of a file (single thread) and report throughput:
//! `cargo run --release --example embed_file -- corpus.txt`.

use std::time::Instant;

use scitok::{Config, Tokenizer};

fn main() {
    let path = std::env::args().nth(1).expect("usage: embed_file <file>");
    let text = std::fs::read_to_string(path).expect("readable file");
    let docs: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let tok = Tokenizer::new(Config::default());
    let start = Instant::now();
    let nnz: usize = docs.iter().map(|d| tok.embed_document(d).indices.len()).sum();
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{} docs, {:.1} MB in {secs:.2}s: {:.1} MB/s, {:.1} µs/doc, {:.1} terms/doc",
        docs.len(),
        text.len() as f64 / 1e6,
        text.len() as f64 / 1e6 / secs,
        secs * 1e6 / docs.len() as f64,
        nnz as f64 / docs.len() as f64
    );
}
