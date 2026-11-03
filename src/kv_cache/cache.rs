//! KV Cache: caches and incrementally reuses the K/V tensors of each attention layer.
//!
//! The cache stores **already-RoPE-rotated** K and unrotated V. This convention comes from the
//! core property of RoPE "attention only depends on relative position": the incremental
//! path rotates the new segment before writing it in, and never re-rotates when reading it out —
//! old K already carries the rotation for its own position, re-rotating it would add a second
//! rotation. llama.cpp's `k_cache` follows the same convention.
//!
//! Per-layer cache shape is `[max_seq_len, num_kv_heads, head_dim]`, and the KV head count
//! follows the model's GQA grouping (for example Llama-7B has 32 query heads but only 8 KV heads,
//! so each KV head is shared by a group of 4 query heads). Under MHA they are equal.
//!
//! Capacity cost: layers × max_seq_len × num_kv_heads × head_dim × 2 (K+V) × 4 bytes.
//! Llama-7B (32 layers, cap 2048) is about 2 GB, so it must be preallocated according to
//! actual context length rather than defaulting to the max value; when generating a small amount
//! on demand, shrinking the capacity to the prompt length + generation amount is more economical.

use crate::tensor::Tensor;

/// Single layer's KV Cache
struct KvCacheLayer {
    /// `[max_seq_len, num_kv_heads, head_dim]`, K with RoPE already applied
    k_cache: Tensor,
    /// `[max_seq_len, num_kv_heads, head_dim]`, raw V, no positional encoding
    v_cache: Tensor,
    /// Number of prefix tokens already written, counting contiguously from position 0
    current_len: usize,
}

/// All layers' KV Cache, each layer has an independent `current_len`
pub struct KvCache {
    layers: Vec<KvCacheLayer>,
    max_seq_len: usize,
    /// KV head count (not query head count under GQA); see `new`
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Create a KV Cache with `num_layers` layers × `[max_seq_len, num_kv_heads, head_dim]`.
    ///
    /// The meaning of `num_kv_heads` is **KV head count**, not the query head count
    /// (`num_attention_heads`) seen in the configuration: what the cache holds is K/V.
    /// Under GQA, passing the query head count in wastes memory (for example
    /// 32 instead of 8, 4x overallocation) and cannot match the shape of the K written by
    /// Attention (`[new_len, num_kv_heads, head_dim]`), causing `update` to be rejected with a shape
    /// error. Call sites should take the value from `TransformerConfig::num_kv_heads`.
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| KvCacheLayer {
                k_cache: Tensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]),
                v_cache: Tensor::zeros(vec![max_seq_len, num_kv_heads, head_dim]),
                current_len: 0,
            })
            .collect();

        Self {
            layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Batch-write the new K/V into layer `layer_idx` at the given absolute positions.
    ///
    /// `k`/`v` must be `[new_len, num_kv_heads, head_dim]`, with head count and head_dim
    /// matching the cache; `positions.len()` must equal `new_len`; positions start contiguously
    /// from the layer's current `current_len`, strictly increasing with no gaps
    /// (`positions[i] == current_len + i`), and each value must be non-negative and less than
    /// `max_seq_len`. Out-of-range, backward, or with gaps are all errors, never silently
    /// truncated or written to a wrapped position. Prefill (start from 0) and decode (continue
    /// after `current_len`) are the same rule.
    ///
    /// `current_len` only advances after the entire batch has been written successfully
    /// (transactional). All validation happens before any write, so a partial failure
    /// mid-batch cannot actually occur; even if a defensive write error fires, because
    /// `current_len` is not advanced, the garbage rows just written are outside the prefix
    /// and forever invisible to `get`.
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        positions: &[i32],
    ) -> Result<(), String> {
        if layer_idx >= self.layers.len() {
            return Err(format!(
                "KV cache layer index {layer_idx} out of range (cache has {} layers)",
                self.layers.len()
            ));
        }
        let layer = &mut self.layers[layer_idx];

        // Shape validation: k/v must match the cache's `[_, num_kv_heads, head_dim]`.
        // Under GQA, a cache wrongly constructed with the query head count is caught here
        // rather than discovering shape mismatch at read time.
        let k_dims = k.dims();
        if k_dims.len() != 3 || k_dims[1] != self.num_kv_heads || k_dims[2] != self.head_dim {
            return Err(format!(
                "K shape {k_dims:?} does not match cache [_, {}, {}]",
                self.num_kv_heads, self.head_dim
            ));
        }
        let v_dims = v.dims();
        if v_dims != k_dims {
            return Err(format!("K/V shape mismatch: K {k_dims:?}, V {v_dims:?}"));
        }
        let new_len = k_dims[0];

        // Position validation, all done before writing:
        // - length must match new_len, otherwise the caller's intent is ambiguous
        // - must continue from current_len (covers backward, gaps, and double-prefill)
        // - out of [0, max_seq_len) is rejected (uses signed comparison to avoid
        //   a negative number becoming a huge number when cast to usize)
        if positions.len() != new_len {
            return Err(format!(
                "K batch is {new_len} token(s) but got {} position(s)",
                positions.len()
            ));
        }
        for (index, &position) in positions.iter().enumerate() {
            let position = usize::try_from(position)
                .map_err(|_| format!("positions[{index}] = {position} is negative"))?;
            let expected = layer.current_len.checked_add(index).ok_or_else(|| {
                format!(
                    "positions[{index}] overflows usize from current_len {}",
                    layer.current_len
                )
            })?;
            if position != expected {
                return Err(format!(
                    "positions[{index}] = {position} must continue from current_len \
                     (expected {expected}); skipping, backward, or double writes are not allowed"
                ));
            }
            if position >= self.max_seq_len {
                return Err(format!(
                    "positions[{index}] = {position} exceeds cache capacity {}",
                    self.max_seq_len
                ));
            }
        }

        for (token, &position) in positions.iter().enumerate() {
            let position = position as usize;
            for head in 0..self.num_kv_heads {
                for dim in 0..self.head_dim {
                    let coords = [token, head, dim];
                    // Shape has been validated above, coordinate reads will not fail;
                    // an internal inconsistency is propagated as an error rather than panicking.
                    let k_value = k.get(&coords).ok_or_else(|| {
                        format!("K is missing coordinate [{token}, {head}, {dim}]")
                    })?;
                    let v_value = v.get(&coords).ok_or_else(|| {
                        format!("V is missing coordinate [{token}, {head}, {dim}]")
                    })?;
                    layer.k_cache.set(&[position, head, dim], k_value)?;
                    layer.v_cache.set(&[position, head, dim], v_value)?;
                }
            }
        }

        // The entire batch has been written, only now advance the length
        if let Some(&last) = positions.last() {
            layer.current_len = layer.current_len.max(last as usize + 1);
        }
        Ok(())
    }

    /// Read out the K/V at the given positions, returning
    /// `[positions.len(), num_kv_heads, head_dim]` tensors.
    ///
    /// Positions must fall in `[0, current_len)`: beyond the written prefix is the content of the
    /// initial zero array, and directly feeding that K/V to attention is equivalent to
    /// fabricating a "token attending to nothing" — so error out instead of silently returning
    /// zeros. Empty `positions` returns a `[0, num_kv_heads, head_dim]` tensor (attention
    /// with an empty batch is a no-op, which is valid).
    ///
    /// The layout is identical to the K/V obtained by the whole-sequence path's
    /// `[seq, num_kv_heads, head_dim]` reshape, so the caller needs no special handling.
    pub fn get(&self, layer_idx: usize, positions: &[i32]) -> Result<(Tensor, Tensor), String> {
        if layer_idx >= self.layers.len() {
            return Err(format!(
                "KV cache layer index {layer_idx} out of range (cache has {} layers)",
                self.layers.len()
            ));
        }
        let layer = &self.layers[layer_idx];

        // Per token read out an entire row (num_kv_heads * head_dim elements), the cache's flat
        // layout is exactly row-major by [position, head, dim], so a contiguous slice copy
        // per position suffices; the boundaries have been checked.
        let row = self.num_kv_heads * self.head_dim;
        let mut k_data = Vec::with_capacity(positions.len() * row);
        let mut v_data = Vec::with_capacity(positions.len() * row);
        for (index, &position) in positions.iter().enumerate() {
            let position = usize::try_from(position)
                .map_err(|_| format!("positions[{index}] = {position} is negative"))?;
            if position >= layer.current_len {
                return Err(format!(
                    "positions[{index}] = {position} is at or beyond the written prefix \
                     (current_len {})",
                    layer.current_len
                ));
            }
            let start = position * row;
            let end = start + row;
            k_data.extend_from_slice(
                layer
                    .k_cache
                    .data()
                    .get(start..end)
                    .ok_or_else(|| format!("K cache is missing rows {start}..{end}"))?,
            );
            v_data.extend_from_slice(
                layer
                    .v_cache
                    .data()
                    .get(start..end)
                    .ok_or_else(|| format!("V cache is missing rows {start}..{end}"))?,
            );
        }

        let dims = vec![positions.len(), self.num_kv_heads, self.head_dim];
        Ok((Tensor::new(k_data, dims.clone()), Tensor::new(v_data, dims)))
    }

    /// Clear all layers' written prefixes, reset `current_len` to 0.
    ///
    /// Only the length is reset, the K/V buffer is not zeroed: after `clear`, `get` rejects
    /// everything with the `current_len` rule, and stale data is unreachable; a subsequent
    /// `update` will overwrite it position by position.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.current_len = 0;
        }
    }

    /// Written prefix length of the first layer.
    ///
    /// Under normal use each layer is updated together in the same order, so this value is
    /// consistent across layers; using it for bounds assertions is more direct than per-layer
    /// lookup. A cache with no layers returns 0.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }
}
