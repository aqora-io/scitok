//! Print the terms of each argument: `cargo run --example tokenize -- "text"`.

use scitok::{Config, Tokenizer};

fn main() {
    let tok = Tokenizer::new(Config::default());
    for text in std::env::args().skip(1) {
        println!("{:?}", tok.tokenize(&text));
    }
}
