"""
Golden GGUF fixture generator for Qllama integration tests.

Produces tests/fixtures/tiny-llama.gguf with a deterministic 2-layer Llama model
using the exact config from the test spec. The forward pass in this file computes
the golden logits used by tests/transformer_from_gguf.rs to validate the Rust
implementation against the llama.cpp specification.

Config (fixed):
    n_embd=16, n_ff=32, n_head=4, n_head_kv=2, n_layer=2,
    n_vocab=64, rope_dim=4, eps=1e-6, theta=10000.0, ctx=128

Weight fill (see fill_matrix / fill_norm_vector):
    Every value comes from one 64-bit LCG, so the file is byte-for-byte
    reproducible. 2-D weights are LCG values scaled by 1/sqrt(in_features),
    landing in roughly [-0.5, 0.5] / sqrt(in_features). 1-D rms_norm weights are
    kept near 1.0, matching real llama.cpp files; this matters because a norm
    weight near 1.0 leaves normalized activations at unit RMS, which is what makes
    attention scores large enough for the RoPE convention to be observable in the
    logits. All 2-D weights are stored as F16 (ggml_type 1) and all 1-D norm
    weights as F32 (ggml_type 0), mirroring real llama.cpp files.

Running this script exits non-zero if the golden logits stop being sensitive to
the RoPE convention, so a future weight-fill change cannot silently turn the
fixture into a test that cannot fail.

Run: python tests/fixtures/generate_tiny_llama.py
Outputs: tests/fixtures/tiny-llama.gguf
"""

from __future__ import annotations

import struct
import math
from typing import Callable, Sequence, Iterable


# === Configuration (from spec) ===
N_EMBD = 16
N_FF = 32
N_HEAD = 4
N_HEAD_KV = 2
N_LAYER = 2
N_VOCAB = 64
ROPE_DIM = 4
EPS = 1e-6
THETA = 10000.0
CTX = 128

# Grouped-query attention: query heads per KV head
GROUP = N_HEAD // N_HEAD_KV
# Head dimension
D_HEAD = N_EMBD // N_HEAD


# === GGUF constants ===
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

# Metadata type tags
META_TYPE_U8 = 0
META_TYPE_I8 = 1
META_TYPE_U16 = 2
META_TYPE_I16 = 3
META_TYPE_U32 = 4
META_TYPE_I32 = 5
META_TYPE_F32 = 6
META_TYPE_BOOL = 7
META_TYPE_STRING = 8
META_TYPE_ARRAY = 9
META_TYPE_U64 = 10
META_TYPE_I64 = 11
META_TYPE_F64 = 12


# === Deterministic weight fill ===
# Linear congruential generator (LCG) with output in [-0.5, 0.5)
# Weights are scaled by 1/sqrt(in_features) for reasonable activation magnitudes.
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
LCG_M = 2**64


def lcg_next(state: int) -> tuple[int, float]:
    """Advance LCG state and return (new_state, value in [-0.5, 0.5))."""
    state = (LCG_A * state + LCG_C) % LCG_M
    value = (state / LCG_M) - 0.5
    return state, value


def fill_matrix(rows: int, cols: int, seed: int) -> list[float]:
    """Fill a logical [rows, cols] matrix (GGUF ne=[cols, rows]) with scaled LCG values.

    Scaling: value * sqrt(1/cols) where cols is the input dimension (ne[0]).
    Stored flat in row-major order: flat[r * cols + c] is element (r, c).
    """
    state = seed
    scale = 1.0 / math.sqrt(cols)
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            state, v = lcg_next(state)
            row.append(v * scale)
        data.extend(row)
    return data


def fill_norm_vector(size: int, seed: int) -> list[float]:
    """Fill a 1-D RMS-norm weight near 1.0 (LCG values mapped to [0.75, 1.25)).

    Real llama.cpp files carry rms_norm weights near 1.0; keeping that property here
    is what makes the fixture's RoPE sensitivity measurable, because a norm weight
    near 1.0 leaves the normalized activations at unit RMS and the attention scores
    large enough for softmax to actually discriminate between positions.
    """
    state = seed
    data = []
    for _ in range(size):
        state, v = lcg_next(state)
        data.append(1.0 + 0.5 * v)
    return data


# Seeds chosen to give distinct weights per tensor
SEED_TOKEN_EMBD = 1
SEED_OUTPUT = 2
SEED_OUTPUT_NORM = 3
SEEDS_PER_LAYER = {
    "attn_q": 10,
    "attn_k": 11,
    "attn_v": 12,
    "attn_output": 13,
    "ffn_gate": 14,
    "ffn_up": 15,
    "ffn_down": 16,
    "attn_norm": 17,
    "ffn_norm": 18,
}


# === GGUF writing utilities ===
def pack_string(s: str) -> bytes:
    """Pack a GGUF string: u64 length + utf8 bytes."""
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_metadata_string(value: str) -> bytes:
    """Pack a metadata string value: type tag + string."""
    return struct.pack("<I", META_TYPE_STRING) + pack_string(value)


def pack_metadata_u32(value: int) -> bytes:
    """Pack a metadata u32 value."""
    return struct.pack("<II", META_TYPE_U32, value)


def pack_metadata_f32(value: float) -> bytes:
    """Pack a metadata f32 value."""
    return struct.pack("<If", META_TYPE_F32, value)


def pack_metadata_array(elem_type: int, elements: Sequence) -> bytes:
    """Pack a metadata array: type tag + elem_type + count + elements."""
    header = struct.pack("<IIQ", META_TYPE_ARRAY, elem_type, len(elements))
    if elem_type == META_TYPE_STRING:
        return header + b"".join(pack_string(e) for e in elements)
    elif elem_type == META_TYPE_U32:
        return header + b"".join(struct.pack("<I", e) for e in elements)
    elif elem_type == META_TYPE_F32:
        return header + b"".join(struct.pack("<f", e) for e in elements)
    else:
        raise ValueError(f"Unsupported array element type: {elem_type}")


def pack_tensor_info(name: str, dims: Sequence[int], ggml_type: int, offset: int) -> bytes:
    """Pack a tensor info: name + n_dims + dims + ggml_type + offset."""
    data = pack_string(name)
    data += struct.pack("<I", len(dims))
    for d in dims:
        data += struct.pack("<Q", d)
    data += struct.pack("<I", ggml_type)
    data += struct.pack("<Q", offset)
    return data


def float_to_f16_bytes(values: Iterable[float]) -> bytes:
    """Encode floats as IEEE binary16 bytes (little-endian), as ggml_type 1 requires.

    struct's "e" format is binary16, so the encoder is the stdlib's rather than a
    hand-rolled bit twiddler. Values beyond the binary16 range saturate to infinity,
    matching ggml's own conversion behaviour.
    """
    result = bytearray()
    for v in values:
        try:
            result.extend(struct.pack("<e", v))
        except OverflowError:
            result.extend(struct.pack("<e", math.inf if v > 0 else -math.inf))
    return bytes(result)


def f16_to_float(data: bytes, offset: int) -> float:
    """Decode one IEEE binary16 value at offset to float32.

    binary16 -> float32 is exact for every bit pattern, so the golden logits see
    precisely the values the Rust loader decodes.
    """
    return struct.unpack_from("<e", data, offset)[0]


def float_to_f32_bytes(values: Iterable[float]) -> bytes:
    """Convert float32 iterable to bytes (little-endian)."""
    return b"".join(struct.pack("<f", v) for v in values)


# === GGUF file construction ===
def build_gguf() -> bytes:
    """Build the complete GGUF file as bytes."""
    metadata: list[tuple[str, bytes]] = []
    tensor_infos: list[tuple[str, list[int], int, int]] = []  # (name, dims, type, offset)
    tensor_data_parts: list[bytes] = []

    # --- Metadata ---
    metadata.append(("general.architecture", pack_metadata_string("llama")))
    metadata.append(("general.name", pack_metadata_string("tiny-llama-golden")))
    metadata.append(("general.file_type", pack_metadata_u32(1)))  # F16

    # Llama config (prefix "llama.")
    metadata.append(("llama.context_length", pack_metadata_u32(CTX)))
    metadata.append(("llama.embedding_length", pack_metadata_u32(N_EMBD)))
    metadata.append(("llama.feed_forward_length", pack_metadata_u32(N_FF)))
    metadata.append(("llama.block_count", pack_metadata_u32(N_LAYER)))
    metadata.append(("llama.attention.head_count", pack_metadata_u32(N_HEAD)))
    metadata.append(("llama.attention.head_count_kv", pack_metadata_u32(N_HEAD_KV)))
    metadata.append(("llama.attention.layer_norm_rms_epsilon", pack_metadata_f32(EPS)))
    metadata.append(("llama.rope.freq_base", pack_metadata_f32(THETA)))
    metadata.append(("llama.rope.dimension_count", pack_metadata_u32(ROPE_DIM)))

    # Tokenizer metadata (required by llama.cpp)
    tokens = [f"tok{i}" for i in range(N_VOCAB)]
    metadata.append(("tokenizer.ggml.tokens", pack_metadata_array(META_TYPE_STRING, tokens)))
    metadata.append(("tokenizer.ggml.model", pack_metadata_string("llama")))

    # Real llama.cpp / gguf-py writers emit metadata keys in sorted order
    metadata.sort(key=lambda kv: kv[0])

    # --- Tensors ---
    def add_matrix(name: str, rows: int, cols: int, seed: int):
        """Add a 2-D tensor (F16) with given logical shape [rows, cols]."""
        data = fill_matrix(rows, cols, seed)
        # GGUF dims: ne[0] = cols (contiguous), ne[1] = rows
        dims = [cols, rows]
        tensor_infos.append((name, dims, GGML_TYPE_F16, None))  # offset filled later
        tensor_data_parts.append(float_to_f16_bytes(data))

    def add_vector(name: str, size: int, seed: int):
        """Add a 1-D tensor (F32)."""
        data = fill_norm_vector(size, seed)
        dims = [size]
        tensor_infos.append((name, dims, GGML_TYPE_F32, None))
        tensor_data_parts.append(float_to_f32_bytes(data))

    # Token embeddings: [n_vocab, n_embd]
    add_matrix("token_embd.weight", N_VOCAB, N_EMBD, SEED_TOKEN_EMBD)

    # Per-layer weights
    for layer in range(N_LAYER):
        base = layer * 100
        # Attention projections (logical shape [n_embd, n_embd] for Q/O, [n_kv_embd, n_embd] for K/V)
        # GGUF convention: ne[0] = in_features (contiguous)
        # Q: [n_embd, n_embd]
        add_matrix(f"blk.{layer}.attn_q.weight", N_EMBD, N_EMBD, base + SEEDS_PER_LAYER["attn_q"])
        # K, V: [n_kv_embd, n_embd] where n_kv_embd = n_head_kv * d_head
        n_kv_embd = N_HEAD_KV * D_HEAD
        add_matrix(f"blk.{layer}.attn_k.weight", n_kv_embd, N_EMBD, base + SEEDS_PER_LAYER["attn_k"])
        add_matrix(f"blk.{layer}.attn_v.weight", n_kv_embd, N_EMBD, base + SEEDS_PER_LAYER["attn_v"])
        # Output projection: [n_embd, n_embd]
        add_matrix(f"blk.{layer}.attn_output.weight", N_EMBD, N_EMBD, base + SEEDS_PER_LAYER["attn_output"])

        # FFN: gate/up [n_ff, n_embd], down [n_embd, n_ff]
        add_matrix(f"blk.{layer}.ffn_gate.weight", N_FF, N_EMBD, base + SEEDS_PER_LAYER["ffn_gate"])
        add_matrix(f"blk.{layer}.ffn_up.weight", N_FF, N_EMBD, base + SEEDS_PER_LAYER["ffn_up"])
        add_matrix(f"blk.{layer}.ffn_down.weight", N_EMBD, N_FF, base + SEEDS_PER_LAYER["ffn_down"])

        # Norms (1-D, F32)
        add_vector(f"blk.{layer}.attn_norm.weight", N_EMBD, base + SEEDS_PER_LAYER["attn_norm"])
        add_vector(f"blk.{layer}.ffn_norm.weight", N_EMBD, base + SEEDS_PER_LAYER["ffn_norm"])

    # Output norm and output weight
    add_vector("output_norm.weight", N_EMBD, SEED_OUTPUT_NORM)
    add_matrix("output.weight", N_VOCAB, N_EMBD, SEED_OUTPUT)

    # --- Compute offsets ---
    # Header: magic(4) + version(4) + tensor_count(8) + metadata_count(8)
    header_size = 4 + 4 + 8 + 8
    # Metadata
    metadata_blob = b""
    for key, value in metadata:
        metadata_blob += pack_string(key) + value
    # Tensor infos
    tensor_info_blob = b""
    for name, dims, ggml_type, _ in tensor_infos:
        tensor_info_blob += pack_tensor_info(name, dims, ggml_type, 0)  # placeholder offset

    # Alignment (default 32)
    alignment = 32
    data_start = header_size + len(metadata_blob) + len(tensor_info_blob)
    padding_to_align = (alignment - (data_start % alignment)) % alignment
    data_start += padding_to_align

    # Assign offsets
    offset = 0
    for i, (name, dims, ggml_type, _) in enumerate(tensor_infos):
        tensor_infos[i] = (name, dims, ggml_type, offset)
        # Compute tensor byte size
        n_elements = 1
        for d in dims:
            n_elements *= d
        if ggml_type == GGML_TYPE_F32:
            offset += n_elements * 4
        elif ggml_type == GGML_TYPE_F16:
            offset += n_elements * 2
        else:
            raise ValueError(f"Unsupported GGML type: {ggml_type}")

    # Rebuild tensor info blob with correct offsets
    tensor_info_blob = b""
    for name, dims, ggml_type, off in tensor_infos:
        tensor_info_blob += pack_tensor_info(name, dims, ggml_type, off)

    # Recompute data_start after offset assignment (should be same)
    data_start = header_size + len(metadata_blob) + len(tensor_info_blob)
    padding_to_align = (alignment - (data_start % alignment)) % alignment
    data_start += padding_to_align

    # --- Assemble file ---
    out = bytearray()
    out.extend(GGUF_MAGIC)
    out.extend(struct.pack("<I", GGUF_VERSION))
    out.extend(struct.pack("<Q", len(tensor_infos)))
    out.extend(struct.pack("<Q", len(metadata)))
    out.extend(metadata_blob)

    # Rebuild tensor infos with final offsets
    tensor_info_blob = b""
    for name, dims, ggml_type, off in tensor_infos:
        tensor_info_blob += pack_tensor_info(name, dims, ggml_type, off)
    out.extend(tensor_info_blob)

    # Padding
    out.extend(b"\x00" * padding_to_align)

    # Tensor data
    for part in tensor_data_parts:
        out.extend(part)

    return bytes(out)


# === Forward pass (reference implementation) ===
def softmax(x: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    max_x = max(x) if x else 0.0
    exp_x = [math.exp(v - max_x) for v in x]
    sum_exp = sum(exp_x)
    return [v / sum_exp for v in exp_x] if sum_exp > 0 else [0.0] * len(x)


def silu(x: float) -> float:
    """SiLU activation: x / (1 + exp(-x))."""
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return x * exp_x / (1.0 + exp_x)


def rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = EPS) -> list[float]:
    """RMS normalization: scale = 1/sqrt(mean(x^2) + eps), out = x * scale * weight."""
    n = len(x)
    if n == 0:
        return []
    mean_sq = sum(v * v for v in x) / n
    scale = 1.0 / math.sqrt(mean_sq + eps)
    return [x[i] * scale * weight[i] for i in range(n)]


def matvec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    """Matrix-vector multiplication: mat is [out_dim][in_dim], vec is [in_dim]."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def apply_rope(vec: list[float], pos: int, d: int, theta: float) -> None:
    """Apply RoPE in-place to the first d elements of vec (GGML_ROPE_TYPE_NORM).

    Pairs (2j, 2j+1) are rotated by angle = pos * theta^(-2j/d).
    This is the "standard" llama.cpp convention for LLM_ARCH_LLAMA.
    """
    for j in range(d // 2):
        freq = theta ** (-2.0 * j / d)
        angle = pos * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        a = vec[2 * j]
        b = vec[2 * j + 1]
        vec[2 * j] = a * cos_a - b * sin_a
        vec[2 * j + 1] = a * sin_a + b * cos_a


def apply_rope_neox(vec: list[float], pos: int, d: int, theta: float) -> None:
    """Apply RoPE using NEOX / rotate_half convention (for sensitivity testing).

    Pairs (j, j + d/2) are rotated instead of (2j, 2j+1).
    """
    half = d // 2
    for j in range(half):
        freq = theta ** (-2.0 * j / d)
        angle = pos * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        a = vec[j]
        b = vec[j + half]
        vec[j] = a * cos_a - b * sin_a
        vec[j + half] = a * sin_a + b * cos_a


class TinyLlama:
    """Minimal Llama model for golden fixture computation."""

    def __init__(self, gguf_bytes: bytes):
        self.tensors: dict[str, list[float]] = {}
        self._parse_gguf(gguf_bytes)

    def _parse_gguf(self, data: bytes) -> None:
        """Parse GGUF and load tensors into float32 format."""
        # Skip header (already validated externally)
        offset = 4 + 4 + 8 + 8  # magic, version, tensor_count, metadata_count

        # Read metadata KV count
        n_meta = struct.unpack_from("<Q", data, 4 + 4 + 8)[0]

        # Skip metadata (we only need tensors)
        for _ in range(n_meta):
            key_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8 + key_len
            type_tag = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            offset = self._skip_metadata_value(data, offset, type_tag)

        # Read tensor count
        n_tensors = struct.unpack_from("<Q", data, 4 + 4)[0]

        # Parse tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            n_dims = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from("<Q", data, offset)[0])
                offset += 8
            ggml_type = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            tensor_offset = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            tensor_infos.append((name, dims, ggml_type, tensor_offset))

        # Alignment
        alignment = 32
        data_start = offset
        padding = (alignment - (data_start % alignment)) % alignment
        data_blob_start = data_start + padding

        # Load tensors
        for name, dims, ggml_type, tensor_off in tensor_infos:
            n_elements = 1
            for d in dims:
                n_elements *= d

            if ggml_type == GGML_TYPE_F32:
                start = data_blob_start + tensor_off
                values = []
                for i in range(n_elements):
                    values.append(struct.unpack_from("<f", data, start + i * 4)[0])
                self.tensors[name] = values
            elif ggml_type == GGML_TYPE_F16:
                start = data_blob_start + tensor_off
                values = [f16_to_float(data, start + i * 2) for i in range(n_elements)]
                self.tensors[name] = values

    def _skip_metadata_value(self, data: bytes, offset: int, type_tag: int) -> int:
        """Skip a metadata value and return the new offset."""
        if type_tag == META_TYPE_U8:
            return offset + 1
        elif type_tag == META_TYPE_I8:
            return offset + 1
        elif type_tag == META_TYPE_U16:
            return offset + 2
        elif type_tag == META_TYPE_I16:
            return offset + 2
        elif type_tag == META_TYPE_U32:
            return offset + 4
        elif type_tag == META_TYPE_I32:
            return offset + 4
        elif type_tag == META_TYPE_F32:
            return offset + 4
        elif type_tag == META_TYPE_BOOL:
            return offset + 1
        elif type_tag == META_TYPE_STRING:
            s_len = struct.unpack_from("<Q", data, offset)[0]
            return offset + 8 + s_len
        elif type_tag == META_TYPE_ARRAY:
            elem_type = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            count = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            for _ in range(count):
                offset = self._skip_metadata_value(data, offset, elem_type)
            return offset
        elif type_tag == META_TYPE_U64:
            return offset + 8
        elif type_tag == META_TYPE_I64:
            return offset + 8
        elif type_tag == META_TYPE_F64:
            return offset + 8
        else:
            raise ValueError(f"Unknown metadata type tag: {type_tag}")

    def _get_matrix(self, name: str, rows: int, cols: int) -> list[list[float]]:
        """Get a 2-D tensor as a list of rows [rows][cols].

        GGUF stores with ne[0] = cols (contiguous), so flat[i] = element (row i//cols, col i%cols).
        We return logical row-major: mat[r][c] = flat[r * cols + c].
        """
        flat = self.tensors[name]
        return [[flat[r * cols + c] for c in range(cols)] for r in range(rows)]

    def _get_vector(self, name: str) -> list[float]:
        """Get a 1-D tensor."""
        return self.tensors[name]

    def forward(self, token_ids: Sequence[int], rope_fn: Callable[[list[float], int, int, float], None] = apply_rope) -> list[list[float]]:
        """Run the Llama forward pass and return logits [seq_len][n_vocab].

        Args:
            token_ids: Input token IDs.
            rope_fn: RoPE function to use (default: standard llama.cpp convention).

        Returns:
            Logits for each position, shape [seq_len, n_vocab].
        """
        seq_len = len(token_ids)
        n_embd = N_EMBD
        n_head = N_HEAD
        n_kv = N_HEAD_KV
        d_head = D_HEAD
        n_ff = N_FF
        group = GROUP
        theta = THETA
        eps = EPS

        # Token embeddings: [n_vocab, n_embd]
        emb_weight = self._get_matrix("token_embd.weight", N_VOCAB, n_embd)
        h = [emb_weight[tid][:] for tid in token_ids]  # [seq, n_embd]

        # Layers
        for layer in range(N_LAYER):
            # --- Attention ---
            attn_norm_w = self._get_vector(f"blk.{layer}.attn_norm.weight")
            # Normalize
            h_norm = [rms_norm(h[i], attn_norm_w, eps) for i in range(seq_len)]

            # Q, K, V projections
            Wq = self._get_matrix(f"blk.{layer}.attn_q.weight", n_embd, n_embd)
            Wk = self._get_matrix(f"blk.{layer}.attn_k.weight", n_kv * d_head, n_embd)
            Wv = self._get_matrix(f"blk.{layer}.attn_v.weight", n_kv * d_head, n_embd)
            Wo = self._get_matrix(f"blk.{layer}.attn_output.weight", n_embd, n_embd)

            q = [matvec(Wq, h_norm[i]) for i in range(seq_len)]  # [seq, n_embd]
            k = [matvec(Wk, h_norm[i]) for i in range(seq_len)]  # [seq, n_kv*d_head]
            v = [matvec(Wv, h_norm[i]) for i in range(seq_len)]  # [seq, n_kv*d_head]

            # Apply RoPE to Q and K (per head, in-place on q and k)
            # Note: Python list slicing creates a copy, so we must write back
            for pos in range(seq_len):
                for hq in range(n_head):
                    qh = q[pos][hq * d_head:(hq + 1) * d_head]
                    rope_fn(qh, pos, ROPE_DIM, theta)
                    # Write back rotated values
                    for j in range(d_head):
                        q[pos][hq * d_head + j] = qh[j]
                for kv_idx in range(n_kv):
                    kh = k[pos][kv_idx * d_head:(kv_idx + 1) * d_head]
                    rope_fn(kh, pos, ROPE_DIM, theta)
                    # Write back rotated values
                    for j in range(d_head):
                        k[pos][kv_idx * d_head + j] = kh[j]

            # Attention output
            attn_out = [[0.0] * n_embd for _ in range(seq_len)]
            scale = 1.0 / math.sqrt(ROPE_DIM)

            for hq in range(n_head):
                kv_idx = hq // group
                # Extract head slices (now containing rotated Q/K)
                qh = [[q[pos][hq * d_head + j] for j in range(d_head)] for pos in range(seq_len)]
                kh = [[k[pos][kv_idx * d_head + j] for j in range(d_head)] for pos in range(seq_len)]
                vh = [[v[pos][kv_idx * d_head + j] for j in range(d_head)] for pos in range(seq_len)]

                # Compute attention for each query position
                for i in range(seq_len):
                    # Compute scores: s[j] = scale * dot(q[i], k[j])
                    scores = []
                    for j in range(seq_len):
                        if j > i:
                            scores.append(float("-inf"))
                        else:
                            dot = sum(qh[i][jj] * kh[j][jj] for jj in range(d_head))
                            scores.append(scale * dot)

                    # Softmax
                    probs = softmax(scores)

                    # Weighted sum of values
                    out_h = [0.0] * d_head
                    for j in range(seq_len):
                        for jj in range(d_head):
                            out_h[jj] += probs[j] * vh[j][jj]

                    # Write back to attention output
                    for jj in range(d_head):
                        attn_out[i][hq * d_head + jj] = out_h[jj]

            # Output projection
            attn_proj = [matvec(Wo, attn_out[i]) for i in range(seq_len)]

            # Residual
            for i in range(seq_len):
                for j in range(n_embd):
                    h[i][j] += attn_proj[i][j]

            # --- FFN ---
            ffn_norm_w = self._get_vector(f"blk.{layer}.ffn_norm.weight")
            h_norm = [rms_norm(h[i], ffn_norm_w, eps) for i in range(seq_len)]

            Wgate = self._get_matrix(f"blk.{layer}.ffn_gate.weight", n_ff, n_embd)
            Wup = self._get_matrix(f"blk.{layer}.ffn_up.weight", n_ff, n_embd)
            Wdown = self._get_matrix(f"blk.{layer}.ffn_down.weight", n_embd, n_ff)

            gate = [[silu(x) for x in matvec(Wgate, h_norm[i])] for i in range(seq_len)]
            up = [matvec(Wup, h_norm[i]) for i in range(seq_len)]
            hidden = [[gate[i][j] * up[i][j] for j in range(n_ff)] for i in range(seq_len)]
            ffn_out = [matvec(Wdown, hidden[i]) for i in range(seq_len)]

            # Residual
            for i in range(seq_len):
                for j in range(n_embd):
                    h[i][j] += ffn_out[i][j]

        # Final norm
        output_norm_w = self._get_vector("output_norm.weight")
        h = [rms_norm(h[i], output_norm_w, eps) for i in range(seq_len)]

        # Output projection
        Wout = self._get_matrix("output.weight", N_VOCAB, n_embd)
        logits = [matvec(Wout, h[i]) for i in range(seq_len)]

        return logits


def main() -> None:
    """Build the fixture and print the golden numbers for tests/transformer_from_gguf.rs."""
    import os

    gguf_bytes = build_gguf()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "tiny-llama.gguf")
    with open(output_path, "wb") as f:
        f.write(gguf_bytes)

    print(f"Wrote {len(gguf_bytes)} bytes to {output_path}")

    model = TinyLlama(gguf_bytes)
    print(f"Loaded {len(model.tensors)} tensors")

    token_ids = [3, 17, 42]
    logits = model.forward(token_ids)
    golden_logits = [round(v, 6) for v in logits[-1]]

    print(f"\nGolden logits for token ids {token_ids} (last position, 6 decimals):")
    print(f"  {golden_logits}")

    print("\nTop 5 tokens:")
    for tok, logit in sorted(enumerate(golden_logits), key=lambda x: -x[1])[:5]:
        print(f"  token {tok}: {logit:.6f}")

    # Prove the fixture can detect a wrong RoPE convention. NEOX pairs (j, j + d/2)
    # instead of (2j, 2j+1), which a correct LLM_ARCH_LLAMA implementation must not do.
    logits_neox = model.forward(token_ids, rope_fn=apply_rope_neox)
    max_diff = max(abs(a - b) for a, b in zip(golden_logits, (round(v, 6) for v in logits_neox[-1])))

    print("\nRoPE sensitivity (NEOX / rotate_half convention):")
    print(f"  Max absolute difference vs golden: {max_diff:.6f}")
    if max_diff < 1e-3:
        raise SystemExit(
            f"FIXTURE HAS NO TEETH: RoPE convention changes logits by only {max_diff:.6f} (< 1e-3). "
            "Pick a different token triple or weight scale."
        )
    print("  Fixture is sensitive to the RoPE convention.")


if __name__ == "__main__":
    main()
