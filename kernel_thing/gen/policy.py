"""
Policy Network - Selects next AST node to emit (CUDA or PTX)

Architecture:
- Small GRU to encode action history
- MLP to predict:
  1. CUDA node type (when not in PTX mode)
  2. PTX node type (when in PTX mode)
  3. Values for the node type

Supports both CUDA AST generation and PTX instruction emission.
This is designed for fast training on 2060 Super.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen.grammar import NodeType, ValueType, NUM_NODE_TYPES, get_node_spec
from gen.ptx_grammar import (
    PTXNodeType, PTXValueType, NUM_PTX_NODE_TYPES, get_ptx_spec,
    PTX_DTYPES, PTX_MODIFIERS
)
from gen.builder_state import BuilderState


# Vocabulary sizes for CUDA value types
CUDA_VOCAB_SIZES = {
    ValueType.IDENTIFIER: 512,
    ValueType.TYPE_NAME: 256,
    ValueType.INTEGER: 128,
    ValueType.STRING: 512,
    ValueType.EXPRESSION: 1024,
    ValueType.QUALIFIER: 8,
    ValueType.STORAGE: 8,
    ValueType.BOOLEAN: 2,
}

# Vocabulary sizes for PTX value types
PTX_VOCAB_SIZES = {
    PTXValueType.REGISTER: 256,     # Register names: %r0-%r255, %rd0, %f0, etc.
    PTXValueType.IMMEDIATE: 256,    # Immediate values
    PTXValueType.MEMORY: 128,       # Memory operands
    PTXValueType.LABEL: 64,         # Labels
    PTXValueType.DTYPE: len(PTX_DTYPES) + 1,
    PTXValueType.MODIFIER: len(PTX_MODIFIERS) + 1,
    PTXValueType.PREDICATE: 32,     # Predicate registers
    PTXValueType.VECTOR: 64,        # Vector operands
}


class PolicyNetwork(nn.Module):
    """
    Policy network for CUDA + PTX AST generation.

    Input: encoded state from BuilderState
    Output:
      - CUDA node_type logits (when not in PTX mode)
      - PTX node_type logits (when in PTX mode)
      - Value logits for each value type
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        num_cuda_types: int = NUM_NODE_TYPES,
        num_ptx_types: int = NUM_PTX_NODE_TYPES,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cuda_types = num_cuda_types
        self.num_ptx_types = num_ptx_types

        # Combined action space for history encoding
        total_types = num_cuda_types + num_ptx_types
        self.total_types = total_types

        # Embeddings
        self.action_embed = nn.Embedding(total_types, hidden_dim // 2)
        self.context_embed = nn.Embedding(num_cuda_types + 1, hidden_dim // 4)
        self.slot_embed = nn.Embedding(16, hidden_dim // 8)
        self.mode_embed = nn.Embedding(2, hidden_dim // 8)  # 0=CUDA, 1=PTX

        # GRU to encode action history
        # Input: action_embed + is_ptx_flag + has_values_flag
        self.gru = nn.GRU(
            input_size=hidden_dim // 2 + 2,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0
        )

        # Combine history with current context
        feature_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8 + hidden_dim // 8 + 4
        self.combine = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # CUDA node type prediction head
        self.cuda_type_head = nn.Linear(hidden_dim, num_cuda_types)

        # PTX node type prediction head
        self.ptx_type_head = nn.Linear(hidden_dim, num_ptx_types)

        # CUDA value prediction heads
        self.cuda_value_heads = nn.ModuleDict({
            vt.name: nn.Linear(hidden_dim, CUDA_VOCAB_SIZES[vt])
            for vt in ValueType if vt != ValueType.NONE
        })

        # PTX value prediction heads
        self.ptx_value_heads = nn.ModuleDict({
            vt.name: nn.Linear(hidden_dim, PTX_VOCAB_SIZES[vt])
            for vt in PTXValueType if vt != PTXValueType.NONE
        })

        # Value critic (for baseline in REINFORCE)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_encoding: Dict[str, Any],
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state_encoding: From BuilderState.encode_state()
            hidden: Optional GRU hidden state

        Returns:
            cuda_logits: [batch, num_cuda_types]
            ptx_logits: [batch, num_ptx_types]
            cuda_value_logits: {value_type_name: [batch, vocab_size]}
            ptx_value_logits: {value_type_name: [batch, vocab_size]}
            value: [batch, 1] - baseline value estimate
            new_hidden: GRU hidden state
        """
        device = next(self.parameters()).device
        batch_size = 1

        # Encode action history
        recent = state_encoding["recent_actions"]
        if len(recent) == 0:
            history_out = torch.zeros(batch_size, self.hidden_dim, device=device)
            new_hidden = None
        else:
            # Build history tensor
            # recent is list of (is_ptx, node_type, has_values)
            action_indices = []
            is_ptx_flags = []
            has_values_flags = []

            for is_ptx, nt, has_vals in recent:
                # Map to combined action space
                if is_ptx:
                    action_idx = self.num_cuda_types + nt
                else:
                    action_idx = nt
                action_indices.append(action_idx)
                is_ptx_flags.append(float(is_ptx))
                has_values_flags.append(float(has_vals))

            action_idx_t = torch.tensor([action_indices], device=device)
            is_ptx_t = torch.tensor([[is_ptx_flags]], device=device).transpose(1, 2)
            has_vals_t = torch.tensor([[has_values_flags]], device=device).transpose(1, 2)

            action_embeds = self.action_embed(action_idx_t)  # [1, seq, dim/2]
            history_input = torch.cat([action_embeds, is_ptx_t, has_vals_t], dim=-1)

            if hidden is None:
                history_out, new_hidden = self.gru(history_input)
            else:
                history_out, new_hidden = self.gru(history_input, hidden)

            history_out = history_out[:, -1, :]

        # Encode current context
        ctx_type = state_encoding["current_context"]
        ctx_idx = torch.tensor([max(0, ctx_type + 1)], device=device)
        ctx_embed = self.context_embed(ctx_idx)

        slot_type = state_encoding["current_slot"]
        slot_idx = torch.tensor([max(0, slot_type + 1)], device=device)
        slot_embed = self.slot_embed(slot_idx)

        # Mode embedding (CUDA vs PTX)
        in_ptx = state_encoding["in_ptx_mode"]
        mode_idx = torch.tensor([1 if in_ptx else 0], device=device)
        mode_embed = self.mode_embed(mode_idx)

        # Additional features
        depth = torch.tensor([[state_encoding["context_depth"] / 10.0]], device=device)
        num_actions = torch.tensor([[state_encoding["num_actions"] / 100.0]], device=device)
        done_flag = torch.tensor([[float(state_encoding["done"])]], device=device)
        ptx_instrs = torch.tensor([[state_encoding.get("ptx_num_instrs", 0) / 50.0]], device=device)

        # Combine everything
        combined = torch.cat([
            history_out,
            ctx_embed,
            slot_embed,
            mode_embed,
            depth,
            num_actions,
            done_flag,
            ptx_instrs,
        ], dim=-1)

        features = self.combine(combined)

        # CUDA node type prediction
        cuda_logits = self.cuda_type_head(features)
        cuda_valid_mask = torch.tensor([state_encoding["valid_mask"]], device=device)
        cuda_logits = cuda_logits.masked_fill(~cuda_valid_mask, float('-inf'))

        # PTX node type prediction
        ptx_logits = self.ptx_type_head(features)
        ptx_valid_mask = torch.tensor([state_encoding["valid_ptx_mask"]], device=device)
        ptx_logits = ptx_logits.masked_fill(~ptx_valid_mask, float('-inf'))

        # CUDA value predictions
        cuda_value_logits = {
            name: head(features)
            for name, head in self.cuda_value_heads.items()
        }

        # PTX value predictions
        ptx_value_logits = {
            name: head(features)
            for name, head in self.ptx_value_heads.items()
        }

        # Value baseline
        value = self.value_head(features)

        return cuda_logits, ptx_logits, cuda_value_logits, ptx_value_logits, value, new_hidden

    def sample_action(
        self,
        state: BuilderState,
        hidden: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[Union[NodeType, PTXNodeType], Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: Current BuilderState
            hidden: GRU hidden state
            temperature: Sampling temperature

        Returns:
            node_type: Sampled NodeType or PTXNodeType
            values: Sampled values dict
            log_prob: Log probability of the action
            new_hidden: Updated GRU hidden state
        """
        encoding = state.encode_state()
        in_ptx_mode = encoding["in_ptx_mode"]

        # Forward pass WITH gradients for log_prob computation
        cuda_logits, ptx_logits, cuda_val_logits, ptx_val_logits, value, new_hidden = \
            self.forward(encoding, hidden)

        if in_ptx_mode:
            # Sample PTX node type
            logits = ptx_logits
            if temperature == 0:
                type_idx = logits.argmax(dim=-1).detach()
            else:
                with torch.no_grad():
                    probs = F.softmax(logits / temperature, dim=-1)
                    type_idx = torch.multinomial(probs, 1).squeeze(-1)

            node_type = PTXNodeType(type_idx.item())
            # Log prob WITH gradient
            log_probs = F.log_softmax(logits, dim=-1)
            type_log_prob = log_probs[0, type_idx]

            # Get PTX values
            spec = get_ptx_spec(node_type)
            values, val_log_prob = self._sample_ptx_values(
                spec, ptx_val_logits, temperature
            )

        else:
            # Sample CUDA node type
            logits = cuda_logits
            if temperature == 0:
                type_idx = logits.argmax(dim=-1).detach()
            else:
                with torch.no_grad():
                    probs = F.softmax(logits / temperature, dim=-1)
                    type_idx = torch.multinomial(probs, 1).squeeze(-1)

            node_type = NodeType(type_idx.item())
            # Log prob WITH gradient
            log_probs = F.log_softmax(logits, dim=-1)
            type_log_prob = log_probs[0, type_idx]

            # Get CUDA values
            spec = get_node_spec(node_type)
            values, val_log_prob = self._sample_cuda_values(
                spec, cuda_val_logits, temperature
            )

        total_log_prob = type_log_prob + val_log_prob
        return node_type, values, total_log_prob, new_hidden

    def _sample_cuda_values(
        self,
        spec,
        value_logits: Dict[str, torch.Tensor],
        temperature: float
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Sample values for a CUDA node type."""
        values = {}
        device = next(self.parameters()).device
        log_probs_list = []

        for val_name, val_type in zip(spec.value_names, spec.value_types):
            if val_type == ValueType.NONE:
                continue

            logits = value_logits[val_type.name]

            if temperature == 0:
                val_idx = logits.argmax(dim=-1).detach()
            else:
                with torch.no_grad():
                    probs = F.softmax(logits / temperature, dim=-1)
                    val_idx = torch.multinomial(probs, 1).squeeze(-1)

            # Log prob WITH gradient
            val_log_probs = F.log_softmax(logits, dim=-1)
            log_probs_list.append(val_log_probs[0, val_idx])

            values[val_name] = self._decode_cuda_value(val_type, val_idx.item())

        if log_probs_list:
            total_log_prob = torch.stack(log_probs_list).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=device, requires_grad=True)

        return values, total_log_prob

    def _sample_ptx_values(
        self,
        spec,
        value_logits: Dict[str, torch.Tensor],
        temperature: float
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Sample values for a PTX node type."""
        values = {}
        device = next(self.parameters()).device
        log_probs_list = []

        for op in spec.operands:
            val_type = op.value_type
            val_name = op.name

            if val_type == PTXValueType.NONE:
                continue

            logits = value_logits[val_type.name]

            if temperature == 0:
                val_idx = logits.argmax(dim=-1).detach()
            else:
                with torch.no_grad():
                    probs = F.softmax(logits / temperature, dim=-1)
                    val_idx = torch.multinomial(probs, 1).squeeze(-1)

            # Log prob WITH gradient
            val_log_probs = F.log_softmax(logits, dim=-1)
            log_probs_list.append(val_log_probs[0, val_idx])

            values[val_name] = self._decode_ptx_value(val_type, val_idx.item())

        if log_probs_list:
            total_log_prob = torch.stack(log_probs_list).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=device, requires_grad=True)

        return values, total_log_prob

    def _decode_cuda_value(self, val_type: ValueType, idx: int) -> Any:
        """Decode a CUDA value index to an actual value."""
        if val_type == ValueType.BOOLEAN:
            return bool(idx)

        if val_type == ValueType.QUALIFIER:
            qualifiers = ["", "__global__", "__device__", "__host__",
                         "__host__ __device__", "__forceinline__",
                         "__noinline__", "inline"]
            return qualifiers[min(idx, len(qualifiers) - 1)]

        if val_type == ValueType.STORAGE:
            storages = ["", "static", "extern", "register", "const",
                       "__shared__", "__constant__", "__restrict__"]
            return storages[min(idx, len(storages) - 1)]

        if val_type == ValueType.INTEGER:
            return idx

        # Placeholder for vocabulary lookup
        return f"${val_type.name}_{idx}"

    def _decode_ptx_value(self, val_type: PTXValueType, idx: int) -> Any:
        """Decode a PTX value index to an actual value."""
        if val_type == PTXValueType.REGISTER:
            # Map to register names based on index
            if idx < 64:
                return f"%r{idx}"
            elif idx < 128:
                return f"%rd{idx - 64}"
            elif idx < 192:
                return f"%f{idx - 128}"
            else:
                return f"%h{idx - 192}"

        if val_type == PTXValueType.IMMEDIATE:
            # Common immediate values
            common = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                     -1, 0xffffffff, 0x80000000]
            if idx < len(common):
                val = common[idx]
                return str(val) if val >= 0 else f"0x{val & 0xffffffff:08x}"
            return str(idx)

        if val_type == PTXValueType.DTYPE:
            if idx < len(PTX_DTYPES):
                return PTX_DTYPES[idx]
            return ".b32"

        if val_type == PTXValueType.MODIFIER:
            if idx < len(PTX_MODIFIERS):
                return PTX_MODIFIERS[idx]
            return ""

        if val_type == PTXValueType.PREDICATE:
            return f"%p{idx}"

        if val_type == PTXValueType.LABEL:
            return f"$L{idx}"

        if val_type == PTXValueType.MEMORY:
            # Memory operands
            if idx < 64:
                return f"[%rd{idx}]"
            return f"[%rd{idx % 64}+{(idx // 64) * 16}]"

        if val_type == PTXValueType.VECTOR:
            # Vector operands
            base = idx * 4
            return f"{{%r{base}, %r{base+1}, %r{base+2}, %r{base+3}}}"

        return f"${val_type.name}_{idx}"


class CUDAVocabulary:
    """Manages CUDA value vocabulary."""

    def __init__(self):
        self.vocabs: Dict[ValueType, Dict[int, str]] = {
            vt: {} for vt in ValueType if vt != ValueType.NONE
        }
        self._init_common()

    def _init_common(self):
        types = ["void", "int", "float", "double", "char", "bool",
                 "int8_t", "int16_t", "int32_t", "int64_t",
                 "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                 "half", "nv_half", "__half", "__half2",
                 "float2", "float4", "int2", "int4",
                 "size_t", "ptrdiff_t", "dim3", "cudaError_t"]
        for i, t in enumerate(types):
            self.vocabs[ValueType.TYPE_NAME][i] = t

        ids = ["threadIdx", "blockIdx", "blockDim", "gridDim",
               "x", "y", "z", "i", "j", "k", "n", "m",
               "tid", "bid", "idx", "offset", "stride",
               "input", "output", "data", "ptr", "buf",
               "temp", "val", "result", "sum", "acc"]
        for i, name in enumerate(ids):
            self.vocabs[ValueType.IDENTIFIER][i] = name

        includes = ["cuda_runtime.h", "cuda.h", "cuda_fp16.h",
                   "mma.h", "cooperative_groups.h", "stdio.h"]
        for i, inc in enumerate(includes):
            self.vocabs[ValueType.STRING][i] = inc

        exprs = ["threadIdx.x", "threadIdx.y", "threadIdx.z",
                 "blockIdx.x", "blockIdx.y", "blockIdx.z",
                 "blockDim.x", "blockDim.y", "blockDim.z",
                 "gridDim.x", "gridDim.y", "gridDim.z",
                 "threadIdx.x + blockIdx.x * blockDim.x",
                 "__syncthreads()", "__syncwarp()",
                 "atomicAdd", "atomicMax", "atomicMin"]
        for i, expr in enumerate(exprs):
            self.vocabs[ValueType.EXPRESSION][i] = expr

    def encode(self, val_type: ValueType, value: str) -> int:
        vocab = self.vocabs[val_type]
        for idx, v in vocab.items():
            if v == value:
                return idx
        idx = len(vocab)
        vocab[idx] = value
        return idx

    def decode(self, val_type: ValueType, idx: int) -> str:
        return self.vocabs[val_type].get(idx, f"UNK_{idx}")


class PTXVocabulary:
    """Manages PTX value vocabulary."""

    def __init__(self):
        self.registers: Dict[int, str] = {}
        self.immediates: Dict[int, str] = {}
        self._init_common()

    def _init_common(self):
        # Pre-populate common registers
        for i in range(64):
            self.registers[i] = f"%r{i}"
        for i in range(32):
            self.registers[64 + i] = f"%rd{i}"
        for i in range(32):
            self.registers[96 + i] = f"%f{i}"

        # Common immediates
        common = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for i, val in enumerate(common):
            self.immediates[i] = str(val)


# Global vocabulary instances
CUDA_VOCAB = CUDAVocabulary()
PTX_VOCAB = PTXVocabulary()


def decode_cuda_values(values: Dict[str, Any], spec) -> Dict[str, Any]:
    """Decode CUDA placeholder values using vocabulary."""
    decoded = {}
    for val_name, val_type in zip(spec.value_names, spec.value_types):
        if val_name not in values:
            continue

        val = values[val_name]
        if isinstance(val, str) and val.startswith("$"):
            # Format: $VALUETYPE_NAME_123 - split on last underscore for index
            content = val[1:]  # Remove $
            last_underscore = content.rfind("_")
            if last_underscore != -1:
                vt_name = content[:last_underscore]
                idx = int(content[last_underscore + 1:])
                vt = ValueType[vt_name]
                decoded[val_name] = CUDA_VOCAB.decode(vt, idx)
            else:
                decoded[val_name] = val
        else:
            decoded[val_name] = val

    return decoded
