#![cfg(feature = "domain-quant-ph")]

use scitok::{Config, Domain, Input, TermDef, Tokenizer, Weighting, term_id};

fn tok() -> Tokenizer {
    Tokenizer::new(Config::default())
}

fn tok_with(set: impl FnOnce(&mut Config)) -> Tokenizer {
    let mut config = Config::default();
    set(&mut config);
    Tokenizer::new(config)
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
    assert_eq!(terms("a $T$-gate"), vec!["~t_gate", "gate"]);
    assert!(has("$\\mathrm{SU}(2)$ symmetry", "su(2)"));
    assert!(has("SU(2) symmetry", "su(2)"));
}

#[test]
fn eponyms_and_hyphens() {
    let t = terms("Lieb–Robinson bounds");
    assert_eq!(t, vec!["~lieb_robinson_bound", "lieb", "robinson", "bound"]);
    assert_eq!(terms("Lieb-Robinson bound"), t);
    assert_eq!(terms("Lieb Robinson bound"), t);
    // joined compounds are opt-in
    let with_compounds = tok_with(|cfg| cfg.options.compounds = true);
    assert!(with_compounds.tokenize("two-qubit gates").contains(&"twoqubit".to_string()));
    assert!(with_compounds.tokenize("non-local").contains(&"nonloc".to_string()));
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
    let value = |t: &str| q.values[q.indices.iter().position(|&i| i == term_id(t)).unwrap()];
    // words count 1, multi-word concepts count concept_weight (0.5 by default)
    assert_eq!((value("surfac"), value("code"), value("~surface_code")), (1.0, 1.0, 0.5));
}

#[test]
fn tf_weighting_counts_terms() {
    let tok = tok_with(|cfg| cfg.weighting = Weighting::Tf);
    let v = tok.embed_document("qubit qubit transmon");
    let qubit = v.indices.iter().position(|&i| i == term_id("qubit")).unwrap();
    assert_eq!(v.values[qubit], 2.0);
}

#[test]
fn full_latex_documents() {
    let tok = tok_with(|cfg| cfg.input = Input::Latex);
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
    let none = tok_with(|cfg| cfg.domains.clear());
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
    let tok = tok_with(|cfg| cfg.domains = vec![a, b, c]);
    assert_eq!(tok.tokenize("widget gadget"), vec!["~curated_b", "~b_gadget"]);
}

#[test]
fn domain_stats_set_avg_len() {
    let stats = "version = 1\ncategory = \"x\"\nfrom = \"2026-01-01\"\nuntil = \"2026-06-01\"\ngenerated_by = \"test\"\ndocs = 10\navg_len = 77.5\n";
    let d = Domain::from_sources("x", None, None, Some(stats), None).unwrap();
    assert_eq!(tok_with(|cfg| cfg.domains = vec![d.clone()]).avg_len(), 77.5);
    let explicit = Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: Some(10.0) };
    let tok = tok_with(|cfg| {
        cfg.domains = vec![d];
        cfg.weighting = explicit;
    });
    assert_eq!(tok.avg_len(), 10.0);
}

#[test]
fn random_inputs_never_panic() {
    const ALPHABET: &[&str] = &[
        "\\", "$", "{", "}", "[", "]", "(", ")", "%", "&", "~", "^", "_", "'", "\"", "-", "–", " ", "\n", ".",
        "a", "Q", "s", "2", "γ", "Ω", "ö", "ﬁ", "’", "\\begin{", "\\end{", "equation}", "figure}", "\\cite{",
        "\\emph", "\\\"", "\\v", "\\text{", "\\mathrm{", "SU(", "QAOA", "(VQE)", "variational quantum ", "'s",
    ];
    let tok = tok();
    let latex = tok_with(|cfg| cfg.input = Input::Latex);
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
        tok.expand(&s);
        latex.expand(&s);
    }
}

#[test]
fn options_toggle_pipeline_parts() {
    use scitok::Options;
    let with = |set: fn(&mut Options)| tok_with(|cfg| set(&mut cfg.options));
    let text = "Lieb-Robinson bounds";
    let no_concepts = with(|o| o.concepts = false).tokenize(text);
    assert_eq!(no_concepts, vec!["lieb", "robinson", "bound"]);
    let phrase_only = with(|o| {
        o.phrase_unigrams = false;
        o.compounds = false;
    });
    assert_eq!(phrase_only.tokenize(text), vec!["~lieb_robinson_bound"]);
    assert_eq!(with(|o| o.stem = false).tokenize("entangled qubits"), vec!["entangled", "qubits"]);
    let weighted = with(|o| o.concept_weight = 2.0).embed_query("QAOA surface code");
    let value = |t: &str| weighted.values[weighted.indices.iter().position(|&i| i == term_id(t)).unwrap()];
    assert_eq!((value("~qaoa"), value("~surface_code")), (1.0, 2.0));
    let dropped = with(|o| o.concept_weight = 0.0).tokenize("QAOA surface code");
    assert_eq!(dropped, vec!["~qaoa", "surfac", "code"]);
}

#[test]
fn acronym_definitions_api() {
    assert_eq!(
        tok().acronym_definitions("noisy intermediate-scale quantum (NISQ) devices"),
        vec![("NISQ".to_string(), "noisy intermediate-scale quantum".to_string())]
    );
}

#[test]
fn clean_text() {
    let text = "$O(\\sqrt{N})$ queries via Grover’s algorithm \\cite{grover96}";
    assert_eq!(tok().clean(text).trim(), "queries via Grover's algorithm");
    let plain = tok_with(|cfg| cfg.input = Input::Plain);
    assert_eq!(plain.clean(text), "$O(\\sqrt{N})$ queries via Grover's algorithm \\cite{grover96}");
}

#[test]
fn expand_names_concepts_once() {
    let expand = |text: &str| tok().expand(text);
    assert_eq!(
        expand("QAOA for MaxCut"),
        "QAOA (quantum approximate optimization algorithm) for MaxCut (maximum cut)"
    );
    assert_eq!(expand("qaoa"), "qaoa (quantum approximate optimization algorithm)");
    assert_eq!(expand("ETH in spin chains"), "ETH (eigenstate thermalization hypothesis) in spin chains");
    assert_eq!(expand("eth in spin chains"), "eth in spin chains");
    assert_eq!(expand("Grover search on qubits"), "Grover search (Grover's algorithm) on qubits");
    assert_eq!(expand("VQE and VQE"), "VQE (variational quantum eigensolver) and VQE");
    assert_eq!(
        expand("VQE-based ansatz, QAOA's depth"),
        "VQE-based (variational quantum eigensolver) ansatz, QAOA's (quantum approximate optimization algorithm) depth"
    );
    assert_eq!(expand("with NISQ."), "with NISQ (noisy intermediate-scale quantum).");
}

#[test]
fn expand_leaves_named_and_defined_concepts() {
    let tok = tok();
    for text in [
        "Lieb-Robinson bounds",
        "the quantum approximate optimization algorithm. QAOA works",
        "the quantum approximate optimisation algorithm (QAOA) works",
    ] {
        assert_eq!(tok.expand(text), text);
    }
    let bare = tok_with(|cfg| cfg.domains.clear());
    assert_eq!(bare.expand("QAOA on $n$ qubits"), bare.clean("QAOA on $n$ qubits"));
}
