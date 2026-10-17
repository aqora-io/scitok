//! Term hashing and weighting into Qdrant sparse vectors.

use rustc_hash::FxHashMap;

/// A Qdrant sparse vector: sorted unique `indices` with their `values`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseVector {
    /// Term ids, see [`term_id`].
    pub indices: Vec<u32>,
    /// One value per index.
    pub values: Vec<f32>,
}

/// How document term frequencies become vector values. Qdrant applies IDF
/// itself when the sparse vector is configured with `modifier: idf`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Weighting {
    /// BM25 term-frequency saturation; `avg_len` defaults to the domain's
    /// average document length.
    Bm25 {
        /// Term-frequency saturation.
        k1: f32,
        /// Document-length normalization.
        b: f32,
        /// Average document length.
        avg_len: Option<f32>,
    },
    /// Raw term counts.
    Tf,
}

impl Default for Weighting {
    fn default() -> Self {
        Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: None }
    }
}

/// Term id: `abs(murmur3_32(term, seed = 0))`, the convention used by
/// fastembed's `Qdrant/bm25`.
pub fn term_id(term: &str) -> u32 {
    (murmur3_32(term.as_bytes(), 0) as i32).unsigned_abs()
}

pub fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = seed;
    let (chunks, tail) = data.as_chunks::<4>();
    for &chunk in chunks {
        let k = u32::from_le_bytes(chunk);
        h ^= k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    if !tail.is_empty() {
        let mut k = 0u32;
        for (i, &b) in tail.iter().enumerate() {
            k |= (b as u32) << (8 * i);
        }
        h ^= k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// Accumulates term counts by id (hash collisions merge), with a per-term
/// value multiplier applied after weighting.
#[derive(Default)]
pub struct Counter {
    counts: FxHashMap<u32, (f32, f32)>,
    total: usize,
}

impl Counter {
    pub fn add(&mut self, term: &str, multiplier: f32) {
        let entry = self.counts.entry(term_id(term)).or_insert((0.0, multiplier));
        entry.0 += 1.0;
        entry.1 = entry.1.max(multiplier);
        self.total += 1;
    }

    pub fn document(self, weighting: Weighting, default_avg_len: f32) -> SparseVector {
        let len = self.total as f32;
        self.finish(|tf, m| {
            m * match weighting {
                Weighting::Tf => tf,
                Weighting::Bm25 { k1, b, avg_len } => {
                    let avg = avg_len.unwrap_or(default_avg_len).max(1.0);
                    tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg))
                }
            }
        })
    }

    pub fn query(self) -> SparseVector {
        self.finish(|_, m| m)
    }

    fn finish(self, weight: impl Fn(f32, f32) -> f32) -> SparseVector {
        let mut pairs: Vec<(u32, (f32, f32))> = self.counts.into_iter().collect();
        pairs.sort_unstable_by_key(|&(i, _)| i);
        SparseVector {
            indices: pairs.iter().map(|&(i, _)| i).collect(),
            values: pairs.iter().map(|&(_, (tf, m))| weight(tf, m)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murmur3_reference_values() {
        // values from Python's mmh3.hash (signed) / the reference implementation
        assert_eq!(murmur3_32(b"", 0), 0);
        assert_eq!(murmur3_32(b"hello", 0), 0x248b_fa47);
        assert_eq!(murmur3_32(b"The quick brown fox jumps over the lazy dog", 0), 0x2e4f_f723);
        assert_eq!(murmur3_32(b"", 1), 0x514e_28b7);
    }

    #[test]
    fn bm25_saturates() {
        let mut c = Counter::default();
        for t in ["a", "a", "b"] {
            c.add(t, 1.0);
        }
        let v = c.document(Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: Some(3.0) }, 100.0);
        let a = v.indices.iter().position(|&i| i == term_id("a")).unwrap();
        let b = v.indices.iter().position(|&i| i == term_id("b")).unwrap();
        assert!((v.values[a] - 2.0 * 2.2 / 3.2).abs() < 1e-6);
        assert!((v.values[b] - 2.2 / 2.2).abs() < 1e-6);
        assert!(v.indices.windows(2).all(|w| w[0] < w[1]));
    }
}
