import pytest

import scitok


@pytest.fixture(scope="module")
def tok():
    return scitok.Tokenizer()


def test_builtin_domains():
    assert "quant-ph" in scitok.builtin_domains()


def test_acronyms_and_long_forms_share_concepts(tok):
    assert "~qaoa" in tok.tokenize("We run QAOA")
    assert "~qaoa" in tok.tokenize("the quantum approximate optimization algorithm")


def test_document_and_query_vectors(tok):
    indices, values = tok.embed_document("surface code surface code thresholds")
    assert indices == sorted(indices)
    assert len(indices) == len(values)
    assert scitok.term_id("~surface_code") in indices
    q_indices, q_values = tok.embed_query("surface code thresholds")
    # words count 1, multi-word concepts count concept_weight (0.5 by default)
    assert set(q_values) == {1.0, 0.5}
    assert set(q_indices) == set(indices)


def test_batch_matches_single(tok):
    texts = ["Lieb-Robinson bounds", "Grover's algorithm on $n$ qubits", ""]
    assert tok.embed_documents(texts) == [tok.embed_document(t) for t in texts]


def test_options():
    plain = scitok.Tokenizer(domains=[], weighting="tf")
    assert plain.lexicon_size == 0
    assert plain.tokenize("QAOA") == ["qaoa"]
    indices, values = plain.embed_document("qubit qubit")
    assert values == [2.0]
    assert scitok.Tokenizer(avg_len=42.0).avg_len == 42.0


def test_clean_and_expand(tok):
    assert tok.clean("Grover's algorithm \\cite{grover96}").strip() == "Grover's algorithm"
    assert tok.expand("QAOA on qubits") == "QAOA (quantum approximate optimization algorithm) on qubits"
    assert scitok.Tokenizer(input="plain").clean("$n$ qubits") == "$n$ qubits"


def test_term_id_matches_fastembed_convention():
    # abs(mmh3.hash("hello")) == 613153351
    assert scitok.term_id("hello") == 613153351


def test_invalid_arguments():
    with pytest.raises(ValueError):
        scitok.Tokenizer(domains=["no-such-domain"])
    with pytest.raises(ValueError):
        scitok.Tokenizer(input="markdown")
    with pytest.raises(ValueError):
        scitok.Tokenizer(domain_paths=["/does/not/exist"])
