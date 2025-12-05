"""
Generation Module - Build CUDA kernels by emitting AST nodes

No templates. No hardcoded parameters. The NN learns everything
from compile/run feedback.

The action space is the CUDA AST grammar itself.

Components:
- grammar.py: NodeType, ValueType, NodeSpec - what the NN can emit
- ast_actions.py: create_node() - instantiate cuda_ast nodes
- builder_state.py: BuilderState - track partial AST during generation
- policy.py: PolicyNetwork - NN that selects next action
- train.py: Trainer - REINFORCE training loop

Usage:
    from gen.train import Trainer, TrainConfig
    from gen.policy import PolicyNetwork
    from task.u8gemv import U8GemvSpec

    task = U8GemvSpec(M=1024, N=1024)
    policy = PolicyNetwork()
    trainer = Trainer(task, policy, TrainConfig())
    best_code, best_reward = trainer.train(num_steps=1000)
"""

# Non-torch imports always available
from gen.grammar import NodeType, ValueType, NUM_NODE_TYPES, get_node_spec, get_valid_next_nodes
from gen.ast_actions import create_node, get_value_names, get_value_types
from gen.builder_state import BuilderState, ContextFrame, SlotType


# Torch-dependent imports via lazy loading
def __getattr__(name):
    if name in ("PolicyNetwork", "VOCAB", "ValueVocabulary", "decode_values"):
        from gen import policy
        return getattr(policy, name)
    elif name in ("Trainer", "TrainConfig", "Episode"):
        from gen import train
        return getattr(train, name)
    raise AttributeError(f"module 'gen' has no attribute '{name}'")
