# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPA factor cache under speculative decoding (OLALA_TPA_FACTOR_SPEC, F1 phase (a)),
CPU only: the switch is opt-in (off: the factor builder still refuses a speculative
config and the page rule is unchanged); on, verify rows are decode rows split exactly
as the Olala token-shift builder splits them, FULL graphs are allowed for uniform
verify batches, each verify token gets one virtual single-token decode row, and the
16-token factor block applies on sm_80 / sm_120 only and makes the K3 light page exact
(capacity design, section 4.2)."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.olala import state as st
from vllm.model_executor.layers.mamba.olala import tpa_factor as tf
from vllm.model_executor.layers.mamba.olala.arch import GpuFamily
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata, MultipleOf
from vllm.v1.attention.backends.olala_diff_tpa_attn import OlalaDiffTPAMetadataBuilder
from vllm.v1.attention.backends.olala_spec_utils import prefill_nosync
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

pytestmark = pytest.mark.cpu_test

DENSE_TOKEN_BYTES = 2 * tf.HKV * tf.HEAD_DIM * 2  # the platform rule's attention page per token
FACTOR_ROW_BYTES = 2 * tf.ROWP * 2


@pytest.fixture
def factor_spec(monkeypatch):
    """Set OLALA_TPA_FACTOR_SPEC (None: unset) and the GPU family seen by tpa_factor."""

    def set_(value: str | None, family: GpuFamily = GpuFamily.SM80):
        if value is None:
            monkeypatch.delenv("OLALA_TPA_FACTOR_SPEC", raising=False)
        else:
            monkeypatch.setenv("OLALA_TPA_FACTOR_SPEC", value)
        monkeypatch.setattr(tf, "gpu_family", lambda device=None: family)
        tf.factor_spec_enabled.cache_clear()

    yield set_
    tf.factor_spec_enabled.cache_clear()


def _vllm_config(k: int | None, mamba_page: int | None = None):
    spec = None if k is None else SimpleNamespace(num_speculative_tokens=k, parallel_drafting=False)
    return SimpleNamespace(
        speculative_config=spec,
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        cache_config=SimpleNamespace(mamba_cache_mode="none", mamba_page_size_padded=mamba_page, block_size=16),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def _factor_builder(cfg):
    spec = FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=tf.ROWP, dtype=torch.bfloat16)
    return tf.OlalaTPAFactorMetadataBuilder(spec, ["model.layers.4.mixer.factor"], cfg, torch.device("cpu"))


def _ts_builder(cfg, k: int):
    spec = MambaSpec(
        block_size=768, shapes=((12, 128), (12, 128)), dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.OLALA_DIFF_TPA, num_speculative_blocks=k,
    )
    return OlalaDiffTPAMetadataBuilder(spec, ["model.layers.4.mixer.shift_state"], cfg, torch.device("cpu"))


def _batch(qlens, ctx, is_prefilling=None, pad_reqs=0, pad_tokens=0, k=3):
    """CommonAttentionMetadata of a reordered batch; ``pad_reqs`` zero-length requests and
    ``pad_tokens`` tokens of FULL CUDA-graph padding as model runner V2 lays them out."""
    n_real = len(qlens)
    num_reqs = n_real + pad_reqs
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
    qsl[1 : n_real + 1] = torch.cumsum(torch.tensor(qlens, dtype=torch.int32), 0)
    qsl[n_real + 1 :] = qsl[n_real]
    seq = torch.zeros(num_reqs, dtype=torch.int32)
    seq[:n_real] = torch.tensor([c + q for c, q in zip(ctx, qlens)], dtype=torch.int32)
    num_tokens = int(qsl[-1]) + pad_tokens
    nblk = 8
    bt = torch.arange(num_reqs * (1 + k) * nblk, dtype=torch.int32).view(num_reqs, -1)
    slots = torch.full((num_tokens,), -1, dtype=torch.int64)
    slots[: int(qsl[-1])] = torch.arange(int(qsl[-1]))
    if is_prefilling is not None:
        pre = torch.zeros(num_reqs, dtype=torch.bool)
        pre[:n_real] = torch.tensor(is_prefilling)
        is_prefilling = pre
    return CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=seq, num_reqs=num_reqs,
        num_actual_tokens=num_tokens, max_query_len=max(qlens), max_seq_len=int(seq.max()),
        block_table_tensor=bt, slot_mapping=slots, is_prefilling=is_prefilling,
        seq_lens_cpu_upper_bound=seq.clone(),
    )


def test_off_by_default_keeps_the_refusal_and_the_page_rule(factor_spec):
    factor_spec(None)
    assert not tf.factor_spec_enabled()
    with pytest.raises(NotImplementedError, match="OLALA_TPA_FACTOR=0"):
        _factor_builder(_vllm_config(k=3))
    for fam in GpuFamily:
        factor_spec(None, fam)
        assert tf.block_align() == tf.TILE
        assert tf.OlalaTPAFactorBackend.get_supported_kernel_block_sizes() == [MultipleOf(tf.TILE)]
    cfg = _vllm_config(k=3)
    assert tf.OlalaTPAFactorMetadataBuilder.get_cudagraph_support(cfg, None) is (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE)


@pytest.mark.parametrize("value", [None, "1"])
def test_no_speculation_is_unchanged(factor_spec, value):
    """Without a speculative config the builder is the single-token decode builder,
    flag or not: a 4-token first request is a prefill, no virtual-row metadata."""
    factor_spec(value)
    cfg = _vllm_config(k=None)
    b = _factor_builder(cfg)
    assert b.reorder_batch_threshold == 1
    assert tf.OlalaTPAFactorMetadataBuilder.get_cudagraph_support(cfg, None) is (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE)
    md = b.build(0, _batch([1, 1, 4], [10, 20, 30]))
    assert (md.num_decodes, md.num_decode_tokens, md.num_prefills, md.num_prefill_tokens) == (2, 2, 1, 4)
    assert md.query_start_loc is None
    md = b.build(0, _batch([4, 1], [10, 20]))
    assert (md.num_decodes, md.num_prefills) == (0, 2)


@pytest.mark.parametrize(
    "family,align",
    [(GpuFamily.SM80, 16), (GpuFamily.SM120, 16), (GpuFamily.SM90, 13), (GpuFamily.SM100, 13),
     (GpuFamily.UNSUPPORTED, 13)],
)
def test_16_token_blocks_on_mma_sync_only(factor_spec, family, align):
    """Hopper / sm_100 (TMA tiles) and every other family (Volta's kernel included)
    keep 13-token blocks."""
    factor_spec("1", family)
    assert tf.block_align() == align
    assert tf.OlalaTPAFactorBackend.get_supported_kernel_block_sizes() == [MultipleOf(align)]


def _mamba_page(k: int) -> int:
    """Raw Mamba-3 page of the light-column layout (K = k) with the folded token shift."""
    dims = dict(tp_world_size=1, num_heads=48, head_dim=64, d_state=128, mimo_dim=4, num_rope_angles=32)
    hf = SimpleNamespace(num_attention_heads=48, num_signal_heads_diff=36, head_dim=128, tpa_rank=4)

    def nbytes(shapes, dtypes):
        return sum(math.prod(s) * torch.empty((), dtype=d).element_size() for s, d in zip(shapes, dtypes))

    m = nbytes(st.mamba3_state_shape(**dims, light_cols=1 + k),
               st.mamba3_state_dtype(torch.bfloat16, "auto", "float32", 1 + k))
    return m + nbytes(*st.folded_token_shift_layout(hf, 1, torch.bfloat16, "auto", 1 + k))


def test_k3_light_page_is_exact_with_16_token_blocks(factor_spec):
    """Platform rule (dense 6144 B/token next to the drafter's 16-token backend): lcm 16
    gives the 304-token block whose page is exactly 768 factor rows; lcm(13, 16) = 208
    gives 416-token pages (2.56 MB)."""
    raw = _mamba_page(3)

    def dense_block(align):
        return align * math.ceil(raw / (align * DENSE_TOKEN_BYTES))

    assert dense_block(math.lcm(16, 16)) == 304
    assert dense_block(math.lcm(16, tf.TILE)) == 416
    page = 304 * DENSE_TOKEN_BYTES
    factor_spec("1", GpuFamily.SM80)
    blk = tf._factor_block_size(_vllm_config(k=3, mamba_page=page))
    assert blk == 768 and blk * FACTOR_ROW_BYTES == page
    factor_spec(None, GpuFamily.SM80)
    blk = tf._factor_block_size(_vllm_config(k=3, mamba_page=page))
    assert blk % tf.TILE == 0 and blk * FACTOR_ROW_BYTES > page


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("value", [None, "1"])
def test_factor_block_is_the_smallest_aligned_cover(factor_spec, k, value):
    factor_spec(value, GpuFamily.SM120)
    align = 16 if value else tf.TILE
    page = 16 * math.ceil(_mamba_page(k) / (16 * DENSE_TOKEN_BYTES)) * DENSE_TOKEN_BYTES
    blk = tf._factor_block_size(_vllm_config(k=k, mamba_page=page))
    assert blk % align == 0
    assert blk * FACTOR_ROW_BYTES >= page > (blk - align) * FACTOR_ROW_BYTES


def test_spec_builder_splits_like_the_token_shift_builder(factor_spec):
    """The factor and token-shift metadata must agree on the verify rows
    (_factor_verify asserts it): pure verify, mixed with a short prefill chunk,
    with and without is_prefilling, and FULL-graph padding."""
    factor_spec("1")
    k = 3
    cfg = _vllm_config(k=k)
    fb, tb = _factor_builder(cfg), _ts_builder(cfg, k)
    assert fb.reorder_batch_threshold == tb.reorder_batch_threshold == 1 + k
    assert tf.OlalaTPAFactorMetadataBuilder.get_cudagraph_support(cfg, None) is AttentionCGSupport.UNIFORM_BATCH
    batches = [  # (batch, (verify reqs, verify tokens, prefill reqs, prefill tokens))
        (_batch([4, 2, 1], [50, 9, 30]), (3, 7, 0, 0)),
        (_batch([4, 2, 1], [50, 9, 30], is_prefilling=[False, False, False]), (3, 7, 0, 0)),
        (_batch([4, 3, 9, 2], [50, 9, 0, 0], is_prefilling=[False, False, True, True]), (2, 7, 2, 11)),
        (_batch([4, 2, 9], [50, 9, 0]), (2, 6, 1, 9)),
        (_batch([4, 4], [17, 33], is_prefilling=[False, False], pad_reqs=2, pad_tokens=8), (4, 16, 0, 0)),
    ]
    for m, want in batches:
        f = fb.build(0, m)
        t = tb.build(0, m)
        assert (f.num_decodes, f.num_decode_tokens, f.num_prefills, f.num_prefill_tokens) == want
        assert (t.num_decodes, t.num_decode_tokens, t.num_prefills, t.num_prefill_tokens) == want
        assert t.spec is not None and t.spec.state_cols.shape[0] == f.num_decodes
        assert f.query_start_loc is m.query_start_loc
        if t.num_prefills:
            assert f.query_start_loc_p_cpu == t.query_start_loc_p_cpu.tolist()


def test_spec_builder_prefill_rows_nosync(factor_spec, monkeypatch):
    """OLALA_PREFILL_NOSYNC: the prefill rows after the verify rows get the same
    metadata from the CPU as from the device."""
    factor_spec("1")
    b = _factor_builder(_vllm_config(k=3))
    m = _batch([4, 2, 9, 5], [50, 9, 0, 12], is_prefilling=[False, False, True, True])
    ref = b.build(0, m)
    monkeypatch.setenv("OLALA_PREFILL_NOSYNC", "1")
    prefill_nosync.cache_clear()
    try:
        got = b.build(0, m)
    finally:
        monkeypatch.delenv("OLALA_PREFILL_NOSYNC")
        prefill_nosync.cache_clear()
    assert (got.num_decodes, got.num_prefills) == (ref.num_decodes, ref.num_prefills) == (2, 2)
    for name in ("query_start_loc_p_cpu", "seq_lens_p_cpu", "has_initial_state_p_cpu", "positions_p_cpu"):
        assert getattr(got, name) == getattr(ref, name), name
    assert ref.positions_p_cpu == [0, 12] and ref.has_initial_state_p_cpu == [False, True]


def _expected_rows(qlens, seq_lens, n_tokens):
    req, sl = [], []
    for r, (q, s) in enumerate(zip(qlens, seq_lens)):
        for t in range(q):
            req.append(r)
            sl.append(s - q + t + 1)
    pad = n_tokens - len(req)
    return req + [0] * pad, sl + [1] * pad


def test_virtual_decode_rows_hand_computed():
    """One single-token row per verify token: request r's block-table row, seq_len
    seq_lens[r] - qlen[r] + t + 1; padding rows past the last request point at
    request 0 with seq_len 1 (the prototype's case, on CPU)."""
    qsl = torch.tensor([0, 4, 5, 8, 9], dtype=torch.int32)  # 3 decode reqs (4, 1, 3), then a prefill
    seq_lens = torch.tensor([100, 7, 30, 50], dtype=torch.int32)
    bt = torch.arange(4 * 6, dtype=torch.int32).view(4, 6)
    md = tf.OlalaTPAFactorMetadata(num_decodes=3, num_prefills=1, num_decode_tokens=10, num_prefill_tokens=1,
                                   num_actual_tokens=9, block_table=bt, seq_lens=seq_lens, slot_mapping=None,
                                   max_seq_len=100, max_query_len=4, query_start_loc=qsl)
    rows, sl = md.virtual_decode_rows()
    assert sl.dtype == torch.int32
    assert sl.tolist() == [97, 98, 99, 100, 7, 28, 29, 30, 1, 1]
    assert torch.equal(rows, bt[torch.tensor([0, 0, 0, 0, 1, 2, 2, 2, 0, 0])])
    assert md.virtual_decode_rows() is md.verify_rows  # built once per step, shared by the layers


@pytest.mark.parametrize("seed", range(6))
def test_virtual_decode_rows_random_padded_batches(seed):
    """Random ragged verify batches in the FULL-graph layout: zero-length lanes
    (in the middle and as padding requests) own no row, padding tokens read
    request 0 with seq_len 1."""
    g = torch.Generator().manual_seed(seed)
    k = int(torch.randint(1, 8, (), generator=g))
    n_real = int(torch.randint(1, 9, (), generator=g))
    pad_reqs = int(torch.randint(0, 4, (), generator=g))
    qlens = torch.randint(0, k + 2, (n_real,), generator=g).clamp_(max=1 + k).tolist()
    qlens[0] = max(qlens[0], 1)
    ctx = torch.randint(0, 3000, (n_real,), generator=g).tolist()
    m = _batch(qlens, ctx, pad_reqs=pad_reqs, pad_tokens=int(torch.randint(0, 6, (), generator=g)), k=k)
    nd, ndt = m.num_reqs, m.num_actual_tokens
    md = tf.OlalaTPAFactorMetadata(num_decodes=nd, num_prefills=0, num_decode_tokens=ndt, num_prefill_tokens=0,
                                   num_actual_tokens=ndt, block_table=m.block_table_tensor, seq_lens=m.seq_lens,
                                   slot_mapping=m.slot_mapping, max_seq_len=m.max_seq_len,
                                   max_query_len=m.max_query_len, query_start_loc=m.query_start_loc)
    rows, sl = md.virtual_decode_rows()
    want_req, want_sl = _expected_rows(qlens, m.seq_lens[:n_real].tolist(), ndt)
    assert sl.tolist() == want_sl
    assert torch.equal(rows, m.block_table_tensor[torch.tensor(want_req, dtype=torch.long)])


def test_verify_splits_follow_the_requests(monkeypatch):
    """The virtual rows of a verify step take the split count of a decode of their requests, not of their
    c (1 + K) rows, with a floor of SPEC_MIN_SPLITS."""
    asked = []
    table = {1: 128, 16: 8, 32: 4, 128: 1}
    monkeypatch.setattr(tf, "num_splits_for_batch", lambda b: asked.append(b) or table[b])
    assert [tf.num_splits_for_verify(n) for n in (1, 16, 32, 128)] == [128, 8, 8, 8]
    assert asked == [1, 16, 32, 128]
    assert tf.SPEC_MIN_SPLITS == 8
