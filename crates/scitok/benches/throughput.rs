//! Throughput on abstract-sized documents. Set `SCITOK_BENCH_CORPUS` to a text
//! file with one document per line to benchmark real data instead.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use scitok::{Config, Tokenizer};

const SAMPLES: &[&str] = &[
    "We study the quantum approximate optimization algorithm (QAOA) for MaxCut on 3-regular graphs. \
     Using matrix product state (MPS) simulations with bond dimension up to $\\chi = 256$, we find that \
     QAOA at depth $p=8$ outperforms the Goemans-Williamson algorithm on instances with $N \\le 60$ qubits.",
    "Fault-tolerant quantum computation requires logical error rates far below physical ones. We benchmark \
     rotated surface codes and bivariate bicycle codes under circuit-level noise with minimum-weight perfect \
     matching and BP-OSD decoders, reporting thresholds near 0.7% and a 10x qubit overhead reduction.",
    "Lieb–Robinson bounds constrain information propagation in lattice systems. We derive a tighter bound for \
     long-range interacting spin chains and apply it to the Heisenberg XXZ model, Schrödinger cat states, \
     and out-of-time-order correlators (OTOCs) measured on a 56-qubit superconducting processor~\\cite{x}.",
    "A variational quantum eigensolver (VQE) with a hardware-efficient ansatz is trained on a trapped-ion \
     device. We mitigate errors with zero-noise extrapolation and observe barren plateaus beyond 20 qubits; \
     VQE energies for H$_2$ and LiH agree with full configuration interaction within chemical accuracy.",
];

fn corpus() -> Vec<String> {
    if let Ok(path) = std::env::var("SCITOK_BENCH_CORPUS") {
        let text = std::fs::read_to_string(path).expect("SCITOK_BENCH_CORPUS readable");
        return text.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
    }
    SAMPLES.iter().cycle().take(2000).map(|s| s.to_string()).collect()
}

fn bench(c: &mut Criterion) {
    let tok = Tokenizer::new(Config::default());
    let docs = corpus();
    let bytes: usize = docs.iter().map(String::len).sum();
    let mut group = c.benchmark_group("embed");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sample_size(20);
    group.bench_function("documents_single_thread", |b| {
        b.iter(|| docs.iter().map(|d| tok.embed_document(d).indices.len()).sum::<usize>())
    });
    group.bench_function("documents_parallel", |b| b.iter(|| tok.embed_documents(&docs).len()));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
