# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine drafter: registration, checkpoint conversion and adjacent injection.

CPU only. The end-to-end drafter is exercised on a GPU by the serving
validation (docs/features/speculative_decoding/dspine.md).
"""

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS
from vllm.transformers_utils.configs.eagle import EAGLEConfig
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import (
    _maybe_override_dspine_config,
)

pytestmark = pytest.mark.cpu_test

NUM_LAYERS = 3


@pytest.fixture(autouse=True)
def single_rank_tp(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


def test_registered_as_dflash_candidate_draft():
    from transformers import Qwen3Config

    assert _SPECULATIVE_DECODING_MODELS["DSpineDraftModel"] == (
        "qwen3_dspine",
        "DSpineQwen3ForCausalLM",
    )
    # EAGLEConfig keeps the architecture instead of prefixing it with DFlash.
    wrapped = EAGLEConfig(
        Qwen3Config(architectures=["DSpineDraftModel"]), method="dflash"
    )
    assert wrapped.architectures == ["DSpineDraftModel"]
    # Candidate-head drafts force Model Runner V2 (the only DSpine speculator).
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            draft_model_config=SimpleNamespace(architectures=["DSpineDraftModel"]),
        )
    )
    assert VllmConfig._is_dflash_candidate_draft(config)


def _original_config():
    """The settings of an original (pre-Speculators) DSpine checkpoint."""
    return SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        block_size=8,
        code_bridge_second_pass={
            "mode": "serial_table",
            "first_exact": NUM_LAYERS - 1,
            "head_topk": 16,
        },
        dflash_config={
            "mask_token_id": 7,
            "target_layer_ids": [1, 2, 3],
            "selected_prefix_code_bridge": {"code_size": 64, "rank": 32, "eps": 1e-6},
            "draft_block_causal": True,
            "selected_prefix_terminal": False,
            "selected_prefix_feature_writes": True,
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_layers": list(range(1, NUM_LAYERS + 1)),
            "selected_prefix_last_write_refine": {
                "kind": "exact_sender",
                "head_topk": 16,
            },
        },
    )


def test_original_checkpoint_config_is_mapped():
    config = _original_config()
    _maybe_override_dspine_config(config)
    draft = config.dflash_config
    assert draft["block_size"] == 8
    assert draft["transfer_dim"] == 64
    assert draft["message_dim"] == 32
    assert draft["transition_top_k"] == 16
    assert draft["transfer_space_eps"] == 1e-6
    assert draft["causal"] is True
    assert draft["sample_from_anchor"] is False


def test_converted_checkpoint_config_is_unchanged():
    config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS,
        dflash_config={"block_size": 8, "transfer_dim": 64, "transition_top_k": 16},
    )
    before = copy.deepcopy(config.dflash_config)
    _maybe_override_dspine_config(config)
    assert config.dflash_config == before


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.dflash_config.update(selected_prefix_feature_write_mode="shared"),
        lambda c: c.dflash_config.update(selected_prefix_feature_layers=[1]),
        lambda c: c.code_bridge_second_pass.update(first_exact=0),
        lambda c: c.code_bridge_second_pass.update(head_topk=8),
        lambda c: setattr(c, "is_causal", False),
    ],
)
def test_unsupported_checkpoint_config_is_rejected(edit):
    config = _original_config()
    edit(config)
    with pytest.raises(ValueError, match="DSpine requires"):
        _maybe_override_dspine_config(config)


def _bridge_holder(transfer_dim=4, message_dim=2):
    from vllm.model_executor.models.qwen3_dspine import DSpineQwen3Model

    holder = SimpleNamespace(
        injection=SimpleNamespace(
            num_layers=NUM_LAYERS,
            transfer_dim=transfer_dim,
            message=SimpleNamespace(output_size=message_dim),
        )
    )
    return holder, DSpineQwen3Model


def _bridge_weights(transfer_dim=4, message_dim=2, hidden=8):
    one = torch.ones(())
    return {
        "draft_block_causal_version": one,
        "last_write_refine_version": one,
        "selected_prefix_bridge.independent_write_version": one,
        "selected_prefix_bridge.terminal_disabled_version": one,
        "selected_prefix_bridge.out.weight": torch.zeros(transfer_dim, message_dim),
        "selected_prefix_bridge.codebook": torch.randn(16, transfer_dim),
        "selected_prefix_bridge.read_weight": torch.randn(transfer_dim, hidden),
        "selected_prefix_bridge.early_read_weights": torch.randn(
            NUM_LAYERS - 1, transfer_dim, hidden
        ),
        "selected_prefix_bridge.feature_write_weights": torch.randn(
            NUM_LAYERS, hidden, message_dim
        ),
        "selected_prefix_bridge.gate.weight": torch.randn(message_dim, transfer_dim),
        "selected_prefix_bridge.gate.bias": torch.randn(message_dim),
        "selected_prefix_bridge.predecessor_gate.weight": torch.randn(
            message_dim, transfer_dim
        ),
        "selected_prefix_bridge.prev.weight": torch.randn(message_dim, transfer_dim),
        "layers.0.mlp.down_proj.weight": torch.randn(hidden, hidden),
    }


def test_bridge_weights_are_mapped_to_injection():
    holder, cls = _bridge_holder()
    weights = _bridge_weights()
    cls._validate_bridge_metadata(holder, weights)
    mapped = dict(cls._map_bridge_weights(holder, weights.items()))
    prefix = "selected_prefix_bridge."
    expected = {
        "transfer_codebook": weights[prefix + "codebook"],
        f"injection.readouts.{NUM_LAYERS - 1}.weight": weights[prefix + "read_weight"],
        "injection.receiver_gate.weight": weights[prefix + "gate.weight"],
        "injection.gate_bias": weights[prefix + "gate.bias"],
        "injection.predecessor_gate.weight": weights[
            prefix + "predecessor_gate.weight"
        ],
        "injection.message.weight": weights[prefix + "prev.weight"],
        "layers.0.mlp.down_proj.weight": weights["layers.0.mlp.down_proj.weight"],
    }
    for i in range(NUM_LAYERS - 1):
        expected[f"injection.readouts.{i}.weight"] = weights[
            prefix + "early_read_weights"
        ][i]
    for i in range(NUM_LAYERS):
        expected[f"injection.writes.{i}.weight"] = weights[
            prefix + "feature_write_weights"
        ][i]
    assert mapped.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(mapped[name], tensor), name


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("last_write_refine_version", torch.full((), 2.0), "checkpoint version"),
        ("selected_prefix_bridge.out.weight", torch.ones(4, 2), "zero terminal"),
    ],
)
def test_unsupported_bridge_metadata_is_rejected(name, value, match):
    holder, cls = _bridge_holder()
    weights = _bridge_weights()
    weights[name] = value
    with pytest.raises(ValueError, match=match):
        cls._validate_bridge_metadata(holder, weights)


def test_wrong_layer_stack_is_rejected():
    holder, cls = _bridge_holder()
    weights = _bridge_weights()
    weights["selected_prefix_bridge.early_read_weights"] = torch.randn(NUM_LAYERS, 4, 8)
    with pytest.raises(ValueError, match="layer stack"):
        list(cls._map_bridge_weights(holder, weights.items()))


def _injection(block_size, hidden=32, transfer_dim=16, message_dim=8):
    from vllm.model_executor.models.qwen3_dspine import AdjacentCausalInjection

    with set_current_vllm_config(VllmConfig()):
        injection = AdjacentCausalInjection(
            hidden_size=hidden,
            transfer_dim=transfer_dim,
            message_dim=message_dim,
            num_layers=NUM_LAYERS,
            block_size=block_size,
            params_dtype=torch.float32,
            quant_config=None,
            eps=1e-6,
            prefix="injection",
        )
    torch.manual_seed(0)
    for param in injection.parameters():
        param.data.normal_(0, 0.3)
    return injection


@pytest.mark.parametrize("num_reqs,block_size", [(1, 4), (3, 4), (2, 8)])
def test_flat_injection_matches_per_block_reference(num_reqs, block_size):
    """The flat form (rows of all blocks at once, a roll for the predecessor)
    equals the per-block definition: position 0 keeps its state, position 1
    takes the anchor token's code as predecessor, position t > 1 the features
    of position t - 1 of the same block."""
    injection = _injection(block_size)
    hidden = 32
    states = torch.randn(num_reqs * block_size, hidden)
    anchor = torch.randn(num_reqs, injection.transfer_dim)
    for layer in range(NUM_LAYERS):
        got = injection(layer, states, anchor_features=anchor)
        for r in range(num_reqs):
            block = states[r * block_size : (r + 1) * block_size]
            features = injection.project(layer, block)
            predecessor = torch.cat((anchor[r : r + 1], features[1:-1]), dim=0)
            want = injection.write(layer, block[1:], features[1:], predecessor)
            out = got[r * block_size : (r + 1) * block_size]
            assert torch.equal(out[0], block[0])
            torch.testing.assert_close(out[1:], want, rtol=1e-5, atol=1e-6)


def test_token_features_scale():
    injection = _injection(4)
    codebook = torch.randn(10, injection.transfer_dim)
    ids = torch.tensor([3, 0, 9])
    feats = injection.token_features(codebook, ids)
    torch.testing.assert_close(feats, codebook[ids] * math.sqrt(injection.transfer_dim))
