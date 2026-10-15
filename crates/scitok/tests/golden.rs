use scitok::{Config, Domain, Input, TermDef, Tokenizer, Weighting, term_id};

fn tok() -> Tokenizer {
    Tokenizer::new(Config::default())
}

fn terms(text: &str) -> Vec<String> {
    tok().tokenize(text)
}

fn has(text: &str, term: &str) -> bool {
    terms(text).iter().any(|t| t == term)
}

#[test]
fn acronym_and_long_form_share_a_concept() {
    assert_eq!(terms("We run QAOA"), vec!["run", "~qaoa"]);
    assert_eq!(
        terms("the quantum approximate optimization algorithm"),
        vec!["~qaoa", "quantum", "approxim", "optim", "algorithm"]
    );
    assert!(has("quantum approximate optimisation algorithms", "~qaoa"));
    assert!(has("qaoa maxcut", "~qaoa"), "lowercase queries still match");
}

#[test]
fn cat_states_vs_cat() {
    assert_eq!(terms("Schr\\\"odinger cat states"), vec!["~cat_state", "schroding", "cat", "state"]);
    assert_eq!(terms("Schrödinger cat states"), terms("Schr\\\"odinger cat states"));
    assert_eq!(terms("the CAT tool"), vec!["cat", "tool"]);
}

#[test]
fn latex_fragments_in_abstracts() {
    assert_eq!(
        terms("$O(\\sqrt{N})$ queries via Grover's algorithm \\cite{grover96}"),
        vec!["queri", "~grover_algorithm", "grover", "algorithm"]
    );
    assert_eq!(terms("a $T$-gate"), vec!["~t_gate", "tgate", "gate"]);
    assert!(has("$\\mathrm{SU}(2)$ symmetry", "su(2)"));
    assert!(has("SU(2) symmetry", "su(2)"));
}

#[test]
fn eponyms_and_hyphens() {
    let t = terms("Lieb–Robinson bounds");
    assert_eq!(t, vec!["~lieb_robinson_bound", "lieb", "liebrobinson", "robinson", "bound"]);
    assert_eq!(terms("Lieb-Robinson bound"), t);
    assert!(has("Lieb Robinson bound", "~lieb_robinson_bound"));
    assert!(has("two-qubit gates", "twoqubit"));
    assert!(has("nonlocal", "nonloc") && has("non-local", "nonloc"));
}

#[test]
fn in_document_acronym_definitions() {
    let t = terms("We introduce the frobnicated quantum widget (FQW). The FQW is fast.");
    let count = |term: &str| t.iter().filter(|x| *x == term).count();
    assert_eq!(count("fqw"), 2, "{t:?}");
    assert_eq!(count("frobnic"), 2, "{t:?}");
    assert_eq!(count("widget"), 2, "{t:?}");
}

#[test]
fn case_sensitive_acronyms() {
    assert_eq!(terms("ETH"), vec!["~eth"]);
    assert_eq!(terms("eth"), vec!["eth"]);
    assert!(has("Eigenstate thermalization hypothesis", "~eth"));
}

#[test]
fn query_and_document_share_indices() {
    let tok = tok();
    for text in [
        "Fault-tolerant surface code thresholds with MWPM decoders",
        "variational quantum eigensolver (VQE) on trapped ions",
        "Lieb-Robinson bound",
    ] {
        assert_eq!(tok.embed_query(text).indices, tok.embed_document(text).indices, "{text}");
    }
}

#[test]
fn query_values_are_one_per_unique_term() {
    let q = tok().embed_query("surface code surface code");
    assert!(q.values.iter().all(|&v| v == 1.0));
    assert!(q.indices.contains(&term_id("~surface_code")));
}

#[test]
fn tf_weighting_counts_terms() {
    let tok = Tokenizer::new(Config { weighting: Weighting::Tf, ..Config::default() });
    let v = tok.embed_document("qubit qubit transmon");
    let qubit = v.indices.iter().position(|&i| i == term_id("qubit")).unwrap();
    assert_eq!(v.values[qubit], 2.0);
}

#[test]
fn full_latex_documents() {
    let tok = Tokenizer::new(Config { input: Input::Latex, ..Config::default() });
    let doc = r"\documentclass{article}
\title{Benchmarking \emph{surface codes}}
\begin{document}
\begin{abstract}
We study the toric code~\cite{kitaev}. See Eq.~\eqref{eq:H}.
\end{abstract}
\begin{equation}H = -\sum_v A_v \label{eq:H}\end{equation}
\end{document}";
    assert_eq!(
        tok.tokenize(doc),
        vec!["benchmark", "~surface_code", "surfac", "code", "~toric_code", "toric", "code"]
    );
}

#[test]
fn domains_change_concepts() {
    let none = Tokenizer::new(Config { domains: vec![], ..Config::default() });
    assert!(!none.tokenize("QAOA").iter().any(|t| t.starts_with('~')));
    assert_eq!(none.tokenize("QAOA"), vec!["qaoa"]);
    assert!(has("QAOA", "~qaoa"));
}

#[test]
fn curated_beats_mined_and_first_domain_wins() {
    let def = |canonical: &str, variant: &str| TermDef {
        canonical: canonical.into(),
        variants: vec![variant.into()],
        case_sensitive: false,
    };
    let a = Domain { name: "a".into(), mined: vec![def("mined_a", "widget")], ..Domain::default() };
    let b = Domain {
        name: "b".into(),
        curated: vec![def("curated_b", "widget"), def("b_gadget", "gadget")],
        ..Domain::default()
    };
    let c = Domain { name: "c".into(), curated: vec![def("c_gadget", "gadget")], ..Domain::default() };
    let tok = Tokenizer::new(Config { domains: vec![a, b, c], ..Config::default() });
    assert_eq!(tok.tokenize("widget gadget"), vec!["~curated_b", "~b_gadget"]);
}

#[test]
fn domain_stats_set_avg_len() {
    let stats = "version = 1\ncategory = \"x\"\nfrom = \"2026-01-01\"\nuntil = \"2026-06-01\"\ngenerated_by = \"test\"\ndocs = 10\navg_len = 77.5\n";
    let d = Domain::from_sources("x", None, None, Some(stats), None).unwrap();
    assert_eq!(Tokenizer::new(Config { domains: vec![d.clone()], ..Config::default() }).avg_len(), 77.5);
    let explicit = Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: Some(10.0) };
    assert_eq!(Tokenizer::new(Config { domains: vec![d], weighting: explicit, ..Config::default() }).avg_len(), 10.0);
}

#[test]
fn random_inputs_never_panic() {
    const ALPHABET: &[&str] = &[
        "\\", "$", "{", "}", "[", "]", "(", ")", "%", "&", "~", "^", "_", "'", "\"", "-", "–", " ", "\n", ".",
        "a", "Q", "s", "2", "γ", "Ω", "ö", "ﬁ", "’", "\\begin{", "\\end{", "equation}", "figure}", "\\cite{",
        "\\emph", "\\\"", "\\v", "\\text{", "\\mathrm{", "SU(", "QAOA", "(VQE)", "variational quantum ", "'s",
    ];
    let tok = tok();
    let latex = Tokenizer::new(Config { input: Input::Latex, ..Config::default() });
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    for _ in 0..20_000 {
        let mut s = String::new();
        for _ in 0..(state % 24) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            s.push_str(ALPHABET[(state % ALPHABET.len() as u64) as usize]);
        }
        tok.embed_document(&s);
        latex.embed_document(&s);
    }
}
