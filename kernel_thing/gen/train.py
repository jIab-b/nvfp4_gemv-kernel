"""
Training Loop - REINFORCE with compile/run/bench feedback

The NN generates kernels by emitting AST nodes.
Feedback comes from:
1. Does it compile? (binary)
2. Does it run without error? (binary)
3. Does it produce correct output? (binary/continuous)
4. How fast is it? (continuous, normalized)

Reward shaping:
- Compile failure: small negative (still learning syntax)
- Runtime error: small negative
- Wrong output: negative proportional to error
- Correct output: positive, scaled by speed
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import sys
from pathlib import Path
import time
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen.grammar import NodeType, get_node_spec
from gen.ptx_grammar import PTXNodeType, get_ptx_spec
from gen.builder_state import BuilderState
from gen.policy import PolicyNetwork, decode_cuda_values, CUDA_VOCAB


@dataclass
class Episode:
    """A single generation episode."""
    states: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Tuple[Union[NodeType, PTXNodeType], Dict[str, Any]]] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    # Final results
    source_code: str = ""
    compiled: bool = False
    ran: bool = False
    correct: bool = False
    speed_gflops: float = 0.0
    reward: float = 0.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # RL
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Generation
    max_actions: int = 2000
    temperature: float = 1.0
    temperature_decay: float = 0.9999

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 10000

    # Output dirs
    compiled_dir: str = "compiled_success"
    bench_dir: str = "success_bench"

    # Reward shaping
    compile_fail_reward: float = -0.1
    runtime_error_reward: float = -0.2
    wrong_output_reward: float = -0.5
    correct_base_reward: float = 1.0
    speed_bonus_scale: float = 1.0


class Trainer:
    """
    REINFORCE trainer for kernel generation.

    Generates kernels, evaluates them, updates policy.
    """

    def __init__(
        self,
        task,  # TaskSpec from task/
        policy: PolicyNetwork,
        config: TrainConfig,
        device: str = "cuda"
    ):
        self.task = task
        self.policy = policy.to(device)
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=config.lr)

        # Import exec modules
        from exec.compile import compile_source, get_sm_version
        from exec.run import load_and_run
        from exec.bench import benchmark_kernel_source
        self.compile_source = compile_source
        self.load_and_run = load_and_run
        self.benchmark_kernel_source = benchmark_kernel_source
        self.sm = get_sm_version()

        # Output dirs
        Path(config.compiled_dir).mkdir(exist_ok=True)
        Path(config.bench_dir).mkdir(exist_ok=True)
        self.compiled_count = 0
        self.bench_count = 0

        # Stats
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.best_code = ""

    def generate_episode(self) -> Episode:
        """Generate one kernel through AST actions (CUDA or PTX)."""
        episode = Episode()
        state = BuilderState()
        hidden = None

        for step in range(self.config.max_actions):
            if state.done:
                break

            # Get state encoding
            encoding = state.encode_state()
            episode.states.append(encoding)

            # Sample action (returns either NodeType or PTXNodeType)
            node_type, values, log_prob, hidden = self.policy.sample_action(
                state, hidden, self.config.temperature
            )

            # Apply action based on mode
            if isinstance(node_type, PTXNodeType):
                # PTX mode - add PTX instruction
                success = state.add_ptx_node(node_type, values)
                decoded_values = values  # PTX values already decoded by policy
            else:
                # CUDA mode - decode and add CUDA node
                spec = get_node_spec(node_type)
                decoded_values = decode_cuda_values(values, spec)
                success = state.add_node(node_type, decoded_values)

            if not success:
                # Invalid action - should not happen with valid masking
                episode.log_probs.append(log_prob)
                episode.actions.append((node_type, decoded_values))
                continue

            episode.actions.append((node_type, decoded_values))
            episode.log_probs.append(log_prob)

            # Get value estimate
            with torch.no_grad():
                result = self.policy.forward(encoding, hidden)
                value = result[4]  # value is 5th return element
            episode.values.append(value)

        # Get generated code
        episode.source_code = state.emit()

        return episode

    def evaluate_episode(self, episode: Episode) -> float:
        """
        Evaluate the generated kernel.

        Returns reward based on:
        1. Compilation success
        2. Runtime success
        3. Correctness
        4. Speed
        """
        code = episode.source_code
        cfg = self.config

        if not code.strip():
            episode.reward = cfg.compile_fail_reward
            return episode.reward

        # Try to compile
        try:
            result = self.compile_source(code, self.sm)
            if not result.success:
                episode.compiled = False
                episode.reward = cfg.compile_fail_reward
                return episode.reward
            episode.compiled = True

            # Save compiled kernel
            self.compiled_count += 1
            compiled_path = Path(cfg.compiled_dir) / f"{self.compiled_count:06d}.cu"
            with open(compiled_path, "w") as f:
                f.write(code)
        except Exception as e:
            episode.compiled = False
            episode.reward = cfg.compile_fail_reward
            return episode.reward

        # Try to run
        try:
            # Generate test input
            inputs = self.task.generate_input()

            # Run kernel via load_and_run (JIT compile + execute)
            run_result = self.load_and_run(code, inputs, sm=self.sm, dtype="uint8")

            if not run_result.success:
                episode.ran = False
                episode.reward = cfg.runtime_error_reward
                return episode.reward

            episode.ran = True
            output = run_result.output

            # Check correctness - reference takes A, B (not C)
            expected = self.task.reference(inputs[0], inputs[1])
            is_correct, msg = self.task.check(output, expected)
            episode.correct = is_correct

            if not is_correct:
                episode.reward = cfg.wrong_output_reward
                return episode.reward

            # Save successful benchmark (correct output)
            self.bench_count += 1
            bench_path = Path(cfg.bench_dir) / f"{self.bench_count:06d}.cu"
            with open(bench_path, "w") as f:
                f.write(f"// Correct output verified\n")
                f.write(code)

            # Benchmark speed
            try:
                # Simple timing - run multiple times
                import torch
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                # Warmup
                for _ in range(3):
                    self.load_and_run(code, inputs, sm=self.sm, dtype="uint8")

                torch.cuda.synchronize()
                start.record()
                for _ in range(10):
                    self.load_and_run(code, inputs, sm=self.sm, dtype="uint8")
                end.record()
                torch.cuda.synchronize()

                ms = start.elapsed_time(end) / 10
                gflops = self.task.flops() / (ms * 1e6)
                episode.speed_gflops = gflops

                # Update bench file with timing
                with open(bench_path, "r") as f:
                    content = f.read()
                with open(bench_path, "w") as f:
                    f.write(f"// GFLOPS: {gflops:.2f}, ms: {ms:.4f}\n")
                    f.write(content)

                # Reward = base + speed bonus
                episode.reward = cfg.correct_base_reward + cfg.speed_bonus_scale * gflops
            except Exception:
                # Correct but couldn't benchmark
                episode.reward = cfg.correct_base_reward

        except Exception as e:
            episode.ran = False
            episode.reward = cfg.runtime_error_reward

        return episode.reward

    def update_policy(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Update policy using collected episodes.

        Uses REINFORCE with baseline.
        """
        cfg = self.config

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for episode in episodes:
            if len(episode.log_probs) == 0:
                continue

            # Compute returns (reward is same for all steps)
            returns = torch.tensor([episode.reward] * len(episode.log_probs),
                                   device=self.device)

            # Stack values
            if len(episode.values) > 0:
                values = torch.stack([v.squeeze() for v in episode.values])
            else:
                values = torch.zeros(len(episode.log_probs), device=self.device)

            # Advantage = return - baseline
            advantages = returns - values.detach()

            # Policy loss (negative because we want to maximize)
            log_probs = torch.stack(episode.log_probs)
            policy_loss = -(log_probs * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus (encourage exploration)
            # Approximate from log_probs
            entropy = -log_probs.mean()

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        # Average over episodes
        n = len(episodes)
        if n == 0:
            return {}

        loss = (
            total_policy_loss / n
            + cfg.value_coef * total_value_loss / n
            - cfg.entropy_coef * total_entropy / n
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": (total_policy_loss / n).item(),
            "value_loss": (total_value_loss / n).item(),
            "entropy": (total_entropy / n).item(),
            "total_loss": loss.item(),
        }

    def train_step(self) -> Dict[str, float]:
        """
        One training step: generate batch, evaluate, update.
        """
        episodes = []

        # Generate batch of episodes
        for _ in range(self.config.batch_size):
            episode = self.generate_episode()
            self.evaluate_episode(episode)
            episodes.append(episode)

            # Track best
            if episode.reward > self.best_reward:
                self.best_reward = episode.reward
                self.best_code = episode.source_code

        # Update policy
        losses = self.update_policy(episodes)

        # Decay temperature
        self.config.temperature *= self.config.temperature_decay

        # Stats
        rewards = [e.reward for e in episodes]
        compiled = sum(e.compiled for e in episodes)
        ran = sum(e.ran for e in episodes)
        correct = sum(e.correct for e in episodes)

        self.episode_count += len(episodes)

        return {
            **losses,
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "compiled_rate": compiled / len(episodes),
            "ran_rate": ran / len(episodes),
            "correct_rate": correct / len(episodes),
            "temperature": self.config.temperature,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
        }

    def train(self, num_steps: int, log_interval: int = 10):
        """
        Full training loop.
        """
        print(f"Starting training for {num_steps} steps")
        print(f"Task: {self.task}")
        print(f"Device: {self.device}")
        print()

        for step in range(num_steps):
            stats = self.train_step()

            if step % log_interval == 0:
                print(f"Step {step:5d} | "
                      f"reward: {stats['mean_reward']:6.3f} "
                      f"(best: {stats['best_reward']:6.3f}) | "
                      f"compiled: {stats['compiled_rate']:.0%} "
                      f"correct: {stats['correct_rate']:.0%} | "
                      f"temp: {stats['temperature']:.4f}")

        print()
        print("Training complete!")
        print(f"Best reward: {self.best_reward}")
        print(f"Best code:")
        print(self.best_code)

        return self.best_code, self.best_reward


def main():
    """Example training run."""
    from task.u8gemv import U8GemvSpec

    # Create task
    task = U8GemvSpec(M=1024, N=1024)

    # Create policy
    policy = PolicyNetwork(hidden_dim=256)

    # Create trainer
    config = TrainConfig(
        batch_size=4,
        max_actions=200,
        lr=1e-3,
    )
    trainer = Trainer(task, policy, config)

    # Train
    best_code, best_reward = trainer.train(num_steps=1000, log_interval=10)


if __name__ == "__main__":
    main()
