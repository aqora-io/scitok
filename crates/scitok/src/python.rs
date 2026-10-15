//! Python bindings (`pip install` via maturin, feature `python`).

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Config, Domain, Input, Options, SparseVector, Tokenizer, Weighting};

type Sparse = (Vec<u32>, Vec<f32>);

fn pair(v: SparseVector) -> Sparse {
    (v.indices, v.values)
}

/// Domain-aware tokenizer producing Qdrant sparse vectors.
///
/// Args:
///     domains: built-in domain names, default ``["quant-ph"]``; ``[]`` for none.
///     domain_paths: directories written by ``scitok-learn``, merged after ``domains``.
///     input: ``"auto"`` (LaTeX pre-pass when ``$`` or ``\`` occurs), ``"plain"`` or ``"latex"``.
///     weighting: ``"bm25"`` or ``"tf"`` for document vectors.
///     k1, b: BM25 parameters.
///     avg_len: BM25 average document length; default from the first domain's stats.
///     concepts, phrase_unigrams, compounds, stem, expand_acronyms, concept_weight:
///         pipeline options (see the Rust ``Options`` docs), mainly for evaluation.
#[pyclass(name = "Tokenizer", module = "scitok", frozen)]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (
        domains=None, domain_paths=None, input="auto", weighting="bm25", k1=1.2, b=0.75, avg_len=None,
        concepts=true, phrase_unigrams=true, compounds=false, stem=true, expand_acronyms=true, concept_weight=0.5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        domains: Option<Vec<String>>,
        domain_paths: Option<Vec<PathBuf>>,
        input: &str,
        weighting: &str,
        k1: f32,
        b: f32,
        avg_len: Option<f32>,
        concepts: bool,
        phrase_unigrams: bool,
        compounds: bool,
        stem: bool,
        expand_acronyms: bool,
        concept_weight: f32,
    ) -> PyResult<Self> {
        let err = |e: crate::Error| PyValueError::new_err(e.0);
        let mut loaded = match domains {
            Some(names) => names.iter().map(|n| Domain::builtin(n)).collect::<Result<Vec<_>, _>>().map_err(err)?,
            None => Config::default().domains,
        };
        for path in domain_paths.unwrap_or_default() {
            loaded.push(Domain::load(path).map_err(err)?);
        }
        let input = match input {
            "auto" => Input::Auto,
            "plain" => Input::Plain,
            "latex" => Input::Latex,
            other => return Err(PyValueError::new_err(format!("input must be auto, plain or latex, not {other:?}"))),
        };
        let weighting = match weighting {
            "bm25" => Weighting::Bm25 { k1, b, avg_len },
            "tf" => Weighting::Tf,
            other => return Err(PyValueError::new_err(format!("weighting must be bm25 or tf, not {other:?}"))),
        };
        let options = Options { concepts, phrase_unigrams, compounds, stem, expand_acronyms, concept_weight };
        Ok(PyTokenizer { inner: Tokenizer::new(Config { domains: loaded, input, weighting, options }) })
    }

    /// Terms a text is indexed under, in order.
    fn tokenize(&self, py: Python<'_>, text: &str) -> Vec<String> {
        py.detach(|| self.inner.tokenize(text))
    }

    /// ``(indices, values)`` for indexing one document.
    fn embed_document(&self, py: Python<'_>, text: &str) -> Sparse {
        py.detach(|| pair(self.inner.embed_document(text)))
    }

    /// ``(indices, values)`` per document, computed in parallel.
    fn embed_documents(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Sparse> {
        py.detach(|| self.inner.embed_documents(&texts).into_iter().map(pair).collect())
    }

    /// ``(indices, values)`` for a search query (weight 1 per unique term).
    fn embed_query(&self, py: Python<'_>, text: &str) -> Sparse {
        py.detach(|| pair(self.inner.embed_query(text)))
    }

    /// In-document acronym definitions as ``[(short, long), ...]``.
    fn acronym_definitions(&self, py: Python<'_>, text: &str) -> Vec<(String, String)> {
        py.detach(|| self.inner.acronym_definitions(text))
    }

    /// Average document length used for BM25.
    #[getter]
    fn avg_len(&self) -> f32 {
        self.inner.avg_len()
    }

    /// Number of concept terms in the merged lexicon.
    #[getter]
    fn lexicon_size(&self) -> usize {
        self.inner.lexicon_len()
    }
}

/// Names of the domains compiled into this build.
#[pyfunction]
fn builtin_domains() -> Vec<&'static str> {
    Domain::builtin_names()
}

/// Sparse index of a term: ``abs(murmur3_32(term))``.
#[pyfunction]
fn term_id(term: &str) -> u32 {
    crate::term_id(term)
}

#[pymodule]
fn scitok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(builtin_domains, m)?)?;
    m.add_function(wrap_pyfunction!(term_id, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
