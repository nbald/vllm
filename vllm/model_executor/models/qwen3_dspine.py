# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    get_masked_input_and_mask,
)

from . import qwen3_dspine_fast as fast_path
from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import WeightsMapper, maybe_prefix

_BRIDGE_METADATA = (
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
    "selected_prefix_bridge.out.weight",
)


class AdjacentCausalInjection(nn.Module):
    """BF16 adjacent injection and FP32 last-write refinement."""

    def __init__(
        self,
        *,
        hidden_size: int,
        transfer_dim: int,
        message_dim: int,
        num_layers: int,
        block_size: int,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transfer_dim = transfer_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.eps = eps

        def linear(input_size, output_size, name):
            return ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, name),
                return_bias=False,
            )

        self.readouts = nn.ModuleList(
            linear(hidden_size, transfer_dim, f"readouts.{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self.writes = nn.ModuleList(
            linear(message_dim, hidden_size, f"writes.{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self.message = linear(transfer_dim, message_dim, "message")
        self.receiver_gate = linear(transfer_dim, message_dim, "receiver_gate")
        self.predecessor_gate = linear(transfer_dim, message_dim, "predecessor_gate")
        self.gate_bias = nn.Parameter(
            torch.empty(message_dim, dtype=params_dtype), requires_grad=False
        )

    @staticmethod
    def rms(states: torch.Tensor) -> torch.Tensor:
        return (
            states.float()
            .square()
            .mean(-1, keepdim=True)
            .clamp_min(torch.finfo(torch.float32).tiny)
            .sqrt()
        )

    def project(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() / (self.rms(hidden_states) + self.eps)
        projected = self.readouts[layer_idx](normalized.to(hidden_states.dtype)).float()
        features = (
            math.sqrt(self.transfer_dim)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        return features.to(hidden_states.dtype)

    def token_features(
        self, codebook: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return (codebook[token_ids.long()].float() * math.sqrt(self.transfer_dim)).to(
            codebook.dtype
        )

    def write(
        self,
        layer_idx,
        receiver_states,
        receiver_features,
        predecessor_features,
        *,
        refine: bool = False,
    ):
        def linear(layer, features):
            if refine:
                return F.linear(features, layer.weight.float())
            return layer(features).float()

        gate = torch.sigmoid(
            self.receiver_gate(receiver_features).float()
            + linear(self.predecessor_gate, predecessor_features)
            + self.gate_bias.float()
        )
        message = linear(self.message, predecessor_features)
        gated_message = gate * message
        if not refine:
            gated_message = gated_message.to(receiver_states.dtype)
        update = linear(self.writes[layer_idx], gated_message)
        result = receiver_states.float() + self.rms(receiver_states) * update
        return result if refine else result.to(receiver_states.dtype)

    def forward(
        self,
        layer_idx: int,
        pre_injection_states: torch.Tensor,
        *,
        anchor_features: torch.Tensor,
    ) -> torch.Tensor:
        if fast_path.FUSED:
            return fast_path.inject(
                self, layer_idx, pre_injection_states, anchor_features
            )
        # Flat over the (requests * block_size) rows, no per-block reshape: rows
        # are independent, position 0 of a block keeps its state and position 1
        # takes the anchor's code as predecessor (same values as the blocked form).
        n = pre_injection_states.shape[0]
        features = self.project(layer_idx, pre_injection_states)
        pos = torch.arange(n, device=pre_injection_states.device) % self.block_size
        predecessor = torch.where(
            (pos == 1)[:, None],
            anchor_features.repeat_interleave(self.block_size, dim=0),
            features.roll(1, dims=0),
        )
        written = self.write(layer_idx, pre_injection_states, features, predecessor)
        return torch.where((pos == 0)[:, None], pre_injection_states, written)


def shared_target_head() -> bool:
    """OLALA_DSPINE_SHARED_HEAD=1: a drafter exported with its own LM head
    (``has_own_lm_head``) uses the target's head instead, whatever its
    quantization (the INT8 W8A16 head of a Q8 target, the bf16 head of a bf16
    one). The own copy is neither allocated past construction nor loaded."""
    return os.environ.get("OLALA_DSPINE_SHARED_HEAD", "0") == "1"


def drop_head_weights(weights: Iterable[tuple[str, torch.Tensor]]):
    """The checkpoint weights without the drafter's own LM head (shared-head
    mode: they are not read into memory, and the loader does not mark the
    drafter as having its own head)."""
    return ((n, w) for n, w in weights if not n.startswith("lm_head."))


def head_rows(lm_head: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """fp32 rows of the LM head for the token ids ``ids``: the floating-point
    weight, or the dequantized rows ``q * scale`` of an INT8 W8A16 head (weight
    ``[V, H]`` int8; ``weight_scale`` ``[V]`` per channel or ``[V, H // g]`` per
    group), i.e. the rows its own GEMM multiplies by."""
    weight = lm_head.weight
    rows = weight[ids].float()
    if weight.is_floating_point():
        return rows
    scale = lm_head.weight_scale
    if scale.dim() == 1:
        return rows * scale[ids].float().unsqueeze(-1)
    group = weight.shape[1] // scale.shape[1]
    return rows * scale[ids].float().repeat_interleave(group, dim=-1)


def _score_transitions(
    injection: AdjacentCausalInjection,
    norm: nn.Module,
    lm_head: ParallelLMHead,
    transfer_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    final_pre_injection: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    final_layer_idx: int,
) -> torch.Tensor:
    top_k = candidate_ids.shape[-1]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    if fast_path.FUSED and lm_head.tp_size == 1:
        return fast_path.score_transitions(
            injection,
            norm,
            lm_head,
            transfer_codebook,
            candidate_ids,
            final_pre_injection,
            predecessor_ids,
            final_layer_idx,
        )
    receiver_features = injection.project(final_layer_idx, final_pre_injection)
    receiver_states = final_pre_injection[:, :, None, :]
    local_ids = candidate_ids.long()
    if lm_head.tp_size > 1:
        shard = lm_head.shard_indices
        local_ids, nonlocal_mask = get_masked_input_and_mask(
            local_ids,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
        )
    successor_weights = head_rows(lm_head, local_ids)
    exact_features = transfer_codebook[predecessor_ids].float() * math.sqrt(
        injection.transfer_dim
    )
    refined = injection.write(
        final_layer_idx,
        receiver_states,
        receiver_features[:, :, None, :],
        exact_features,
        refine=True,
    )
    normalized = (
        refined
        / (refined.square().mean(-1, keepdim=True) + norm.variance_epsilon).sqrt()
        * norm.weight.float()
    )
    scores = torch.einsum("bnkh,bnjh->bnkj", normalized, successor_weights)
    if lm_head.tp_size > 1:
        scores.masked_fill_(nonlocal_mask.unsqueeze(-2), 0)
        scores = tensor_model_parallel_all_reduce(scores)
    return scores


# The draft step runs inside a FULL CUDA graph; torch.compile of the injection
# loop spent >10 min in symbolic-shape guard simplification for little gain.
@ignore_torch_compile
class DSpineQwen3Model(DFlashQwen3Model):
    hf_to_vllm_mapper = DFlashQwen3Model.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_substr={"transfer_embeddings": "transfer_codebook"},
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        block_size = 1 + speculative_config.num_speculative_tokens
        configured_block_size = int(draft_config.get("block_size", block_size))
        # Draft blocks are causal (attention and adjacent injection), so a shorter
        # block is an exact prefix of a trained one: fewer speculative tokens than
        # block_size - 1 are served unchanged.
        if block_size > configured_block_size:
            raise ValueError(
                "DSpine needs 1 + num_speculative_tokens <= block_size: "
                f"{block_size} > {configured_block_size}."
            )
        transfer_dim = int(draft_config["transfer_dim"])
        if transfer_dim > self.config.hidden_size:
            raise ValueError("DSpine transfer_dim cannot exceed hidden_size.")
        self.transition_top_k = int(draft_config["transition_top_k"])
        self.injection = AdjacentCausalInjection(
            hidden_size=self.config.hidden_size,
            transfer_dim=transfer_dim,
            message_dim=int(draft_config["message_dim"]),
            num_layers=len(self.layers),
            block_size=block_size,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            eps=float(draft_config.get("transfer_space_eps", 1e-6)),
            prefix=maybe_prefix(prefix, "injection"),
        )
        self.transfer_codebook = nn.Parameter(
            torch.empty(
                self.config.vocab_size,
                transfer_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None and fast_path.hand_prep_ok(self):
            # OLALA_DSPINE_FUSED=3: embeddings and anchor features in one launch
            input_embeds, anchor_features = fast_path.hand_prep(self, input_ids)
        else:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            anchor_features = self.injection.token_features(
                self.transfer_codebook, input_ids[:: self.injection.block_size]
            )
        hidden_states = input_embeds
        residual = None
        final_pre_injection = hidden_states
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            pre_injection = (
                hidden_states if residual is None else hidden_states + residual
            )
            hidden_states = self.injection(
                layer_idx, pre_injection, anchor_features=anchor_features
            )
            if layer_idx == len(self.layers) - 1:
                final_pre_injection = pre_injection
            residual = None
        hidden_states = self.norm(hidden_states)
        return hidden_states, final_pre_injection

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        # OLALA_DSPINE_FUSED=3: projection, then norm + rotary + every layer's cache write in one launch
        if not fast_path.hand_context_kv(
            self, context_states, context_positions, context_slot_mapping
        ):
            super().precompute_and_store_context_kv(
                context_states, context_positions, context_slot_mapping
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        if not any(name.startswith("selected_prefix_bridge.") for name, _ in weights):
            return super().load_weights(weights)
        self._validate_bridge_metadata(dict(weights))
        mapped = list(self._map_bridge_weights(weights))
        loaded = super().load_weights(mapped)
        required = {"transfer_codebook"} | {
            f"injection.{name}" for name, _ in self.injection.named_parameters()
        }
        if missing := required - loaded:
            raise ValueError(f"Missing DSpine bridge weights: {sorted(missing)}")
        return loaded

    def _validate_bridge_metadata(self, weights: dict[str, torch.Tensor]) -> None:
        if missing := set(_BRIDGE_METADATA[2:]) - weights.keys():
            raise ValueError(f"Missing DSpine bridge metadata: {sorted(missing)}")
        for name in _BRIDGE_METADATA[:-1]:
            if name in weights:
                weight = weights[name]
                if weight.numel() != 1 or weight.item() != 1:
                    raise ValueError(f"Unsupported DSpine checkpoint version: {name}")
        terminal = weights["selected_prefix_bridge.out.weight"]
        if (
            tuple(terminal.shape)
            != (self.injection.transfer_dim, self.injection.message.output_size)
            or terminal.count_nonzero().item()
        ):
            raise ValueError("DSpine requires a zero terminal projection")

    def _map_bridge_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        aliases = {
            "codebook": "transfer_codebook",
            "read_weight": f"injection.readouts.{self.injection.num_layers - 1}.weight",
            "gate.weight": "injection.receiver_gate.weight",
            "gate.bias": "injection.gate_bias",
            "predecessor_gate.weight": "injection.predecessor_gate.weight",
            "prev.weight": "injection.message.weight",
        }
        for name, weight in weights:
            if name in _BRIDGE_METADATA:
                continue
            key = name.removeprefix("selected_prefix_bridge.")
            if key in ("early_read_weights", "feature_write_weights"):
                read = key == "early_read_weights"
                count = self.injection.num_layers - int(read)
                if weight.ndim != 3 or weight.shape[0] != count:
                    raise ValueError(f"Invalid DSpine layer stack: {name}")
                for index, tensor in enumerate(weight):
                    module = "readouts" if read else "writes"
                    yield f"injection.{module}.{index}.weight", tensor
            else:
                yield aliases.get(key, name), weight


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DSpineQwen3Model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if self.shares_target_head:
            # The target's head replaces this one after loading (load_dflash_model).
            weights = drop_head_weights(weights)
        loaded = super().load_weights(weights)
        # Opt-in fast draft path (OLALA_DSPINE_* flags, qwen3_dspine_fast.py).
        fast_path.apply_fast_path(self)
        return loaded

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError("DSpine requires the full target vocabulary.")
        # OLALA_DSPINE_SHARED_HEAD: DSpineSpeculator turned has_own_lm_head off;
        # free the unquantized placeholder head now (load_dflash_model installs
        # the target's).
        self.shares_target_head = shared_target_head() and not getattr(
            self.config, "has_own_lm_head", False
        )
        if self.shares_target_head:
            del self.lm_head

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.transition_top_k
        )

    def compute_transition_scores(
        self,
        candidate_ids: torch.Tensor,
        final_pre_injection: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _score_transitions(
            self.model.injection,
            self.model.norm,
            self.lm_head,
            self.model.transfer_codebook,
            candidate_ids,
            final_pre_injection,
            anchor_token_ids,
            len(self.model.layers) - 1,
        )


EntryClass = DSpineQwen3ForCausalLM
