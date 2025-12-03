# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple RL training loop with GRPO-style advantage estimation.

This demonstrates:
1. Loading a model in TorchTitan format for training
2. Converting weights to vLLM format for fast rollouts
3. Generating samples using vLLM
4. Computing rewards (trivial/random for now)
5. Computing advantages using GRPO-style group ranking
6. Performing a policy gradient update on TorchTitan model
7. Optional real dataset support (GSM8K math dataset)
8. Distributed training with TorchTitan parallelisms (TP, FSDP, DDP)

Training modes (all use RayTrainWorker with torchtitan):
- Single GPU: Set tp=1, fsdp=1, ddp=1 (1 worker, no parallelism)
- Distributed: Set tp/fsdp/ddp > 1 (multiple workers with TP/FSDP/DDP)

Example configs:
- Single GPU:      tp=1, fsdp=1, ddp=1  (1 GPU)
- Tensor Parallel: tp=2, fsdp=1, ddp=1  (2 GPUs, model split)
- FSDP:            tp=1, fsdp=2, ddp=1  (2 GPUs, params sharded)
- Combined:        tp=2, fsdp=2, ddp=1  (4 GPUs, TP + FSDP)
"""

import os
import re

import ray

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter

from torchtitan.experiments.deterministic_vllm_rl.weights.converter import (
    torchtitan_to_vllm,
    vllm_to_torchtitan,
)
from torchtitan.experiments.deterministic_vllm_rl.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)

from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from transformers import AutoConfig, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.batch_invariant import init_batch_invariance

init_batch_invariance()


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    Create a StatelessProcessGroup for weight updates.

    vLLM provides StatelessProcessGroup to create a process group
    without interfering with the global process group in torch.distributed.
    This is necessary because vLLM workers already have their own process group
    for tensor parallelism.

    Args:
        master_address: IP address of rank 0 process
        master_port: Port for rendezvous
        rank: Rank of this process in the weight update group
        world_size: Total number of processes in the group
        device: torch.device for NCCL operations

    Returns:
        PyNcclCommunicator instance for NCCL operations
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    # print(f"DEBUG: process group rank={rank}/{world_size}")
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        """
        Initialize weight update group for this vLLM worker.

        Args:
            master_address: IP address of rank 0 (training process)
            master_port: Port for rendezvous
            rank_offset: Offset to add to this worker's rank
            world_size: Total number of processes (trainer + all vLLM workers)
        """
        from vllm.distributed.parallel_state import get_world_group
        # Get this worker's rank within vLLM's TP group and add offset
        rank = get_world_group().rank + rank_offset

        # Create StatelessProcessGroup for weight updates
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        print(f"Worker initialized weight update group: rank={rank}/{world_size}")

    def recv_weights(self, name, dtype_name, shape):
        """
        Receive a single weight tensor via NCCL broadcast.

        Args:
            name: Weight parameter name (e.g., "model.layers.0.mlp.gate_up_proj")
            dtype_name: Data type as string ("float32", "float16", "bfloat16")
            shape: Tuple of tensor dimensions
        """
        # Convert dtype string to torch dtype
        dtype = getattr(torch, dtype_name)

        # Allocate buffer and receive from trainer (src=0)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        # Load into vLLM model
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


@ray.remote
class VLLMRolloutEngine:
    """
    vLLM engine for fast rollouts with weight updates.

    Note: vLLM loads from model_config.model path, so we create a temporary
    directory with updated weights and restart the engine. This is faster than
    recreating temp dirs repeatedly and handles config/tokenizer files properly.

    Args:
        model_path: Path to HuggingFace model (for config/tokenizer)
        temp_checkpoint_dir: Directory to save temporary weight checkpoints
        tp_size: Tensor parallel size (number of GPUs)
    """

    def __init__(
        self,
        model_path: str,
        temp_checkpoint_dir: str = "./converted",
        tp_size: int = 1,
    ):
        self.base_model_path = model_path
        self.tp_size = tp_size
        self.temp_model_dir = os.path.abspath(
            os.path.join(temp_checkpoint_dir, "vllm_temp_model")
        )
        os.makedirs(self.temp_model_dir, exist_ok=True)

        import glob

        # Copy config/tokenizer files from base model to temp dir
        import shutil

        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ]:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, self.temp_model_dir)

        # Copy the original model shard files if they exist
        # We'll overwrite these with our single model.safetensors later
        for shard_file in glob.glob(os.path.join(model_path, "model-*.safetensors")):
            dst = os.path.join(self.temp_model_dir, os.path.basename(shard_file))
            shutil.copy2(shard_file, dst)

        # Copy index file if it exists
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            shutil.copy2(index_file, self.temp_model_dir)

        self.llm = LLM(
            model=self.temp_model_dir,
            trust_remote_code=True,
            max_model_len=2048,
            dtype="bfloat16",
            gpu_memory_utilization=0.3,  # Reduced from 0.5
            seed=42,  # Fixed seed for determinism
            enforce_eager=True,
            distributed_executor_backend="ray",
            tensor_parallel_size=self.tp_size,
            worker_extension_cls="torchtitan.experiments.wide_ep_rl.simple_rl_ray.WorkerExtension",
        )

    def update_weights(self, vllm_compat_state: dict) -> None:
        """
        Update vLLM model weights from vLLM-compat state dict.

        This converts weights to vLLM format, saves them, and reloads using
        vLLM's reload_weights() API after updating the model path config.

        Args:
            vllm_compat_state: vLLM-compat model state dict (with gate_up_proj/down_proj)
        """
        # Convert vLLM-compat -> vLLM (torchtitan_to_vllm handles both formats)
        vllm_state = torchtitan_to_vllm(vllm_compat_state)

        # Save to temp model directory
        checkpoint_path = os.path.join(self.temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        import glob
        import json

        shard_files = sorted(
            glob.glob(os.path.join(self.temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self.temp_model_dir, "model.safetensors.index.json")

        if len(shard_files) == 2 and os.path.exists(index_file):
            # Load the index to see which weights go in which shard
            with open(index_file, "r") as f:
                index_data = json.load(f)

            weight_map = index_data["weight_map"]

            # Split weights according to the index
            shard1_weights = {}
            shard2_weights = {}

            for key, value in vllm_state.items():
                shard_file = weight_map.get(key, shard_files[0])
                if "model-00001-of-00002" in shard_file:
                    shard1_weights[key] = value
                else:
                    shard2_weights[key] = value

            # Ensure weights stay in bfloat16
            shard1_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard1_weights.items()
            }
            shard2_weights = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in shard2_weights.items()
            }

            # Save to the shard files
            save_file(shard1_weights, shard_files[0])
            save_file(shard2_weights, shard_files[1])
        else:
            # Ensure weights stay in bfloat16
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            # Fallback: save as single file
            save_file(vllm_state, checkpoint_path)

        # First time: create the engine
        if self.llm is None:
            self.llm = LLM(
                model=self.temp_model_dir,
                trust_remote_code=True,
                max_model_len=2048,
                dtype="bfloat16",
                gpu_memory_utilization=0.3,  # Reduced from 0.5
                seed=42,  # Fixed seed for determinism
                enforce_eager=True,
                tensor_parallel_size=self.tp_size,
                worker_extension_cls="torchtitan.experiments.wide_ep_rl.simple_rl_ray.WorkerExtension",
            )
            print("✓ Created new vLLM engine")
        else:
            # Use collective_rpc to call reload_weights on all workers
            # This reloads weights from temp_model_dir without recreating the engine
            self.llm.collective_rpc("reload_weights")

    @torch.no_grad()
    def generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        n_samples_per_prompt: int = 4,
    ) -> tuple[
        list[str], torch.Tensor, list[list[int]], list[list[float]], list[list[int]]
    ]:
        """
        Generate samples using vLLM.

        Args:
            prompt_texts: List of prompt strings
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            completions: List of completion strings
            log_probs: [batch] - Sum of log probs for each completion
            token_ids: List of token ID lists for each completion (generated tokens only)
            token_log_probs: List of per-token log prob lists for each completion
            prompt_token_ids: List of prompt token ID lists for each completion
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
        )

        outputs = self.llm.generate(prompt_texts, sampling_params)

        # Extract completions and log probs
        completions = []
        log_probs_list = []
        token_ids_list = []
        token_log_probs_list = []
        prompt_token_ids_list = []

        for output in outputs:
            # Extract prompt token IDs from the output
            prompt_token_ids = output.prompt_token_ids

            for sample in output.outputs:
                completions.append(sample.text)

                # Store prompt tokens for this sample
                prompt_token_ids_list.append(prompt_token_ids)

                # Extract token IDs (generated tokens only)
                token_ids = sample.token_ids
                token_ids_list.append(token_ids)

                # Extract per-token log probs
                per_token_log_probs = [
                    list(logprob_dict.values())[0].logprob
                    for logprob_dict in sample.logprobs
                ]
                token_log_probs_list.append(per_token_log_probs)

                # Sum log probs across generated tokens
                total_log_prob = sum(per_token_log_probs)
                log_probs_list.append(total_log_prob)

        log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

        return (
            completions,
            log_probs,
            token_ids_list,
            token_log_probs_list,
            prompt_token_ids_list,
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()

    def recv_weights(self, name, dtype_name, shape):
        """Wrapper to call recv_weights on all vLLM workers via collective RPC."""
        return self.llm.collective_rpc("recv_weights", args=(name, dtype_name, shape))

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        """Wrapper to initialize weight update group on all vLLM workers."""
        return self.llm.collective_rpc(
            "init_weight_update_group",
            args=(master_address, master_port, rank_offset, world_size),
        )


def dtype_to_str(dtype):
    if dtype == torch.float32:
        return "float32"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class NoOpDataLoader:
    """Minimal no-op dataloader for RL training (no dataset needed)."""

    def __iter__(self):
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


def build_noop_dataloader(**kwargs):
    """No-op dataloader builder for RL training."""
    return NoOpDataLoader()


@ray.remote
class RayTrainWorker:
    """Distributed training worker using torchtitan parallelisms (TP, FSDP, DDP)."""

    def __init__(self, job_config, world_size, rank, master_addr, master_port):
        import os as _os

        import torchtitan.protocols.train_spec as train_spec_module
        from torchtitan.train import Trainer

        # Store os module for later use
        self.os = _os

        # TODO: investigate local ranks.
        # Set environment variables for torch.distributed
        _os.environ["WORLD_SIZE"] = str(world_size)
        _os.environ["RANK"] = str(rank)
        _os.environ["LOCAL_RANK"] = "0"  # Each Ray actor gets its own GPU
        _os.environ["MASTER_ADDR"] = master_addr
        _os.environ["MASTER_PORT"] = str(master_port)

        # Store job config
        self.job_config = job_config

        # Patch train spec to use no-op dataloader (RL doesn't need dataloaders)
        original_get_train_spec = train_spec_module.get_train_spec

        def patched_get_train_spec(model_name):
            spec = original_get_train_spec(model_name)
            spec.build_dataloader_fn = build_noop_dataloader
            return spec

        train_spec_module.get_train_spec = patched_get_train_spec

        # Initialize parent Trainer
        self.trainer = Trainer(self.job_config)
        self.trainer.checkpointer.load()
        self.device = self.trainer.device
        self.model_update_group = None

        # Save initial weights (clone to CPU for tracking)
        self.initial_state = self._get_model_state_dict(clone=True, to_cpu=True)

        print(f"RayTrainWorker initialized: rank={rank}/{world_size}")

    def _get_model_state_dict(self, clone: bool = False, to_cpu: bool = False):
        """
        Get model state dict, handling both FSDP and TP.

        Args:
            clone: If True, clone tensors (for tracking weight changes)
            to_cpu: If True, move tensors to CPU (for tracking weight changes)

        Returns:
            State dict with model parameters
        """
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )

        model = self.trainer.model_parts[0]
        # Use PyTorch DCP's get_model_state_dict which handles both FSDP and TP
        # This will properly gather full state across TP ranks on rank 0
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,  # Gather full model across TP/FSDP
                cpu_offload=to_cpu,  # Offload to CPU if requested
            ),
        )

        # Apply clone if requested (CPU offload is already handled above)
        if clone:
            result = {}
            for k, v in state_dict.items():
                result[k] = v.clone()
            return result
        else:
            return state_dict

    def init_weight_update_group(self, master_address, master_port, world_size):
        """Initialize NCCL for weight broadcasting."""
        rank = int(self.os.environ["RANK"])
        self.model_update_group = stateless_init_process_group(
            master_address=master_address,
            master_port=master_port,
            rank=0,  # Trainer is always rank 0
            world_size=world_size,
            device=self.device,
        )
        print(f"Trainer initialized weight update group: rank=0/{world_size}")

    def broadcast_weights(self, vllm_engine):
        # """Broadcast model weights to vLLM workers."""
        # assert (
        #     self.model_update_group is not None or self.os.environ["RANK"] != 0
        # ), "Call init_weight_update_group first"
        # print("DEBUG: broadcast_weights")
        # titan_state = self._get_model_state_dict()

        # if self.os.environ["RANK"] == 0:
        #     vllm_compat_state = torchtitan_to_vllm(titan_state)
        #     for name, tensor in vllm_compat_state.items():
        #         dtype_name = dtype_to_str(tensor.dtype)
        #         shape = tensor.shape
        #         tensor = tensor.to(self.device)

        #         handle = vllm_engine.recv_weights.remote(name, dtype_name, shape)
        #         self.model_update_group.broadcast(
        #             tensor, src=0, stream=torch.cuda.current_stream()
        #         )
        #         ray.get(handle)


        from torch.distributed.tensor import DTensor

        rank = self.os.environ["RANK"]
        params = self.trainer.model_parts[0].state_dict()
        for name, param in params.items():
            dtype_name = dtype_to_str(param.dtype)
            shape = param.shape
            param = param.to(self.device).full_tensor() if isinstance(param, DTensor) else param
            if rank == 0:
                handle = vllm_engine.recv_weights.remote(name, dtype_name, shape)
                torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
                ray.get(handle)
            torch.distributed.barrier()

    def forward_backward(
        self,
        vllm_token_ids,
        vllm_token_log_probs,
        prompt_token_ids,
        advantages,
        num_rollout_batches,
        kl_coef=0.1,
        ppo_clip_eps=0.2,
        entropy_coef=0.01,
    ):
        """Compute RL loss and backward pass."""
        model = self.trainer.model_parts[0]

        #TODO:current simple fsdp sharding logic assumes tp ==1, need to handle tp > 1, 
        # Shard inputs for FSDP (assuming tp=1)
        rank = int(self.os.environ["RANK"])
        world_size = int(self.os.environ["WORLD_SIZE"])
        
        batch_size = len(vllm_token_ids)
        shard_size = batch_size // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank < world_size - 1 else batch_size
        
        # Shard list inputs by slicing
        vllm_token_ids_shard = vllm_token_ids[start_idx:end_idx]
        vllm_token_log_probs_shard = vllm_token_log_probs[start_idx:end_idx]
        prompt_token_ids_shard = prompt_token_ids[start_idx:end_idx]
        
        # Shard advantages tensor across batch dimension
        advantages_shard = advantages[start_idx:end_idx]

        # Compute loss with distributed training contexts
        with self.trainer.train_context(None):
            with self.trainer.maybe_enable_amp:
                loss, metrics = compute_policy_gradient_loss_vllm(
                    model,
                    vllm_token_ids_shard,
                    vllm_token_log_probs_shard,
                    prompt_token_ids_shard,
                    advantages_shard,
                    kl_coef=kl_coef,
                    ppo_clip_eps=ppo_clip_eps,
                    entropy_coef=entropy_coef,
                )
            
            # Scale loss for gradient accumulation
            loss = loss / num_rollout_batches
            loss.backward()

        return loss.item(), metrics

    def optimizer_step(self):
        """Update weights with gradient clipping."""
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for m in self.trainer.model_parts for p in m.parameters()],
            max_norm=1.0,
        )

        # Update weights
        self.trainer.optimizers.step()
        self.trainer.optimizers.zero_grad()

    def compute_weight_deltas(self):
        """Compute weight changes from initial state."""
        deltas = {}
        module_stats = {}

        with torch.no_grad():
            # Get current state (clone to CPU for comparison with initial state)
            current_state = self._get_model_state_dict(clone=True, to_cpu=True)

            for name, current_param in current_state.items():
                if name not in self.initial_state:
                    continue

                initial_param = self.initial_state[name]
                delta = current_param - initial_param

                parts = name.split(".")
                module_name = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

                delta_norm = torch.linalg.vector_norm(delta).item()
                param_norm = torch.linalg.vector_norm(current_param).item()
                relative_change = delta_norm / (param_norm + 1e-8)

                if module_name not in module_stats:
                    module_stats[module_name] = {"norms": [], "relative": []}

                module_stats[module_name]["norms"].append(delta_norm)
                module_stats[module_name]["relative"].append(relative_change)

            for module_name, stats in module_stats.items():
                deltas[f"weight_delta/{module_name}/magnitude"] = sum(
                    stats["norms"]
                ) / len(stats["norms"])
                deltas[f"weight_delta/{module_name}/relative_change"] = sum(
                    stats["relative"]
                ) / len(stats["relative"])

        return deltas


class TrainGroup:
    """
    Orchestrator for distributed training workers.

    This class spawns RayTrainWorker actors with torchtitan parallelisms (TP, FSDP, DDP).
    Works for both single-GPU (tp=1, fsdp=1, ddp=1) and multi-GPU setups.

    The orchestrator provides a simple unified API (broadcast_weights, forward_backward,
    optimizer_step) that works the same whether you're using 1 GPU or 100 GPUs.
    """

    def __init__(
        self,
        titan_checkpoint_path,
        model_path,
        use_vllm_compat,
        learning_rate,
        tp=1,
        fsdp=1,
        ddp=1,
    ):
        """
        Initialize training group with torchtitan parallelisms.

        Args:
            titan_checkpoint_path: Path to model checkpoint (unused, config comes from torchtitan)
            model_path: Path to HF model
            use_vllm_compat: Unused (torchtitan handles model creation)
            learning_rate: Learning rate
            tp: Tensor parallel degree (default: 1 = no TP)
            fsdp: FSDP degree (default: 1 = no FSDP)
            ddp: DDP degree (default: 1 = no DDP)
        """
        self.tp = tp
        self.fsdp = fsdp
        self.ddp = ddp
        self.num_workers = tp * fsdp * ddp
        self.learning_rate = learning_rate

        # Always use RayTrainWorker (handles both single and multi-GPU)
        if self.num_workers == 1:
            print("Using single GPU training (RayTrainWorker, no parallelism)")
        else:
            print(
                f"Using distributed training: TP={tp}, FSDP={fsdp}, DDP={ddp} ({self.num_workers} GPUs)"
            )

        self.workers = self._create_distributed_workers(
            titan_checkpoint_path, model_path, learning_rate
        )

    def _create_distributed_workers(
        self, titan_checkpoint_path, model_path, learning_rate
    ):
        """Create distributed training workers with torchtitan config."""
        import socket

        from torchtitan.config.job_config import (
            Checkpoint,
            Comm,
            Debug,
            Experimental,
            FaultTolerance,
            Job,
            JobConfig,
            LRScheduler,
            Model,
            Optimizer,
            Parallelism,
            Profiling,
            Training,
            Validation,
        )
        from vllm.utils.network_utils import get_ip

        # Get master address and port
        master_addr = get_ip()

        def get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                return s.getsockname()[1]

        master_port = get_free_port()

        # Create minimal torchtitan config (hardcoded for simplicity)
        job_config = JobConfig(
            job=Job(
                description=f"RL training TP={self.tp} FSDP={self.fsdp}",
                dump_folder="/tmp/torchtitan_rl",
            ),
            model=Model(
                name="qwen3",
                flavor="1.7B",
                hf_assets_path=model_path,
            ),
            training=Training(
                dtype="bfloat16",
                steps=1000000,
                local_batch_size=1,
                global_batch_size=self.num_workers,
                seq_len=2048,
                max_norm=1.0,
            ),
            optimizer=Optimizer(
                name="AdamW",
                lr=learning_rate,
            ),
            lr_scheduler=LRScheduler(
                warmup_steps=0,
            ),
            parallelism=Parallelism(
                tensor_parallel_degree=self.tp,
                data_parallel_shard_degree=self.fsdp,
                data_parallel_replicate_degree=self.ddp,
            ),
            checkpoint=Checkpoint(
                enable=True,
                initial_load_path=titan_checkpoint_path,  # Load model from checkpoint
                initial_load_in_hf=True,
                initial_load_model_only=True,  # Only load model weights, not optimizer/scheduler
                load_only=True,
            ),
            validation=Validation(enable=False),
        )

        # Create Ray actors for each rank
        workers = []
        for rank in range(self.num_workers):
            worker = RayTrainWorker.options(
                num_gpus=1,
                name=f"train_worker_{rank}",
            ).remote(
                job_config=job_config,
                world_size=self.num_workers,
                rank=rank,
                master_addr=master_addr,
                master_port=master_port,
            )
            workers.append(worker)

        # Wait for initialization
        ray.get([w.__ray_ready__.remote() for w in workers])
        print(f"✓ Initialized {self.num_workers} distributed workers")

        return workers

    def init_weight_update_group(self, master_address, master_port, world_size):
        """Initialize weight update group (only rank 0 worker)."""
        # Only rank 0 needs to join the weight update group for broadcasting to vLLM
        ray.get(
            self.workers[0].init_weight_update_group.remote(
                master_address, master_port, world_size
            )
        )

    def broadcast_weights(self, vllm_engine):
        """Broadcast weights from rank 0 to vLLM."""
        # Only rank 0 broadcasts (whether single or multi-GPU)
        handles = [w.broadcast_weights.remote(vllm_engine) for w in self.workers]
        ray.get(handles)

    def forward_backward(self, *args, **kwargs):
        """Forward+backward on all workers."""
        handles = [w.forward_backward.remote(*args, **kwargs) for w in self.workers]
        results = ray.get(handles)
        # Return results from first worker
        return results[0]

    def optimizer_step(self):
        """Optimizer step on all workers."""
        handles = [w.optimizer_step.remote() for w in self.workers]
        results = ray.get(handles)
        return results[0]  # Return grad norm from first worker

    def compute_weight_deltas(self):
        """Compute weight deltas from first worker."""
        return ray.get(self.workers[0].compute_weight_deltas.remote())


def rl_update_step(
    train_group,
    vllm_engine,
    tokenizer,
    prompt_texts: list[str],
    expected_answers: list[str] | None = None,
    group_size: int = 8,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    num_rollout_batches: int = 1,
    use_stable_grpo: bool = False,
    grpo_beta: float = 0.1,
    reward_fn=None,
):
    """
    Perform one RL update step using Ray for orchestration and NCCL for weight sync.

    Args:
        train_group: Ray actor reference for TrainGroup
        vllm_engine: Ray actor reference for VLLMRolloutEngine
        tokenizer: Tokenizer for reward computation
        prompt_texts: List of prompt strings
        expected_answers: List of expected answers for each prompt
        group_size: Number of samples per prompt for GRPO
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        num_rollout_batches: Number of rollout batches per update
        use_stable_grpo: If True, use stable GRPO (mean-centering)
        grpo_beta: Beta parameter for GRPO exponential weighting
        reward_fn: Reward function (defaults to trivial_reward_function)

    Returns:
        metrics: Dict of training metrics
    """
    # Default reward function
    if reward_fn is None:
        reward_fn = trivial_reward_function

    # Broadcast updated weights to vLLM workers via NCCL
    train_group.broadcast_weights(vllm_engine)

    # Multiply prompt_texts by num_rollout_batches to generate all samples at once
    expanded_prompt_texts = prompt_texts * num_rollout_batches
    expanded_expected_answers = (
        expected_answers * num_rollout_batches if expected_answers else None
    )

    # Generate all samples in one call
    (
        completions,
        vllm_log_probs,
        vllm_token_ids,
        vllm_token_log_probs,
        prompt_token_ids,
    ) = ray.get(
        vllm_engine.generate.remote(
            expanded_prompt_texts,
            max_new_tokens,
            temperature,
            n_samples_per_prompt=group_size,
        )
    )
    all_completions = []
    all_rewards = []
    all_advantages = []
    total_loss = 0.0
    batch_metrics = []

    # Number of completions per batch
    samples_per_batch = len(prompt_texts) * group_size

    for batch_idx in range(num_rollout_batches):
        # Slice the results for this batch
        start_idx = batch_idx * samples_per_batch
        end_idx = start_idx + samples_per_batch

        batch_completions = completions[start_idx:end_idx]
        batch_vllm_token_ids = vllm_token_ids[start_idx:end_idx]
        batch_vllm_token_log_probs = vllm_token_log_probs[start_idx:end_idx]
        batch_prompt_token_ids = prompt_token_ids[start_idx:end_idx]

        # Compute rewards using provided reward function
        if reward_fn == trivial_reward_function:
            rewards = reward_fn(
                batch_completions, tokenizer, expected_answers, group_size
            )
        elif reward_fn == math_reward_function:
            rewards = reward_fn(batch_completions, expected_answers, group_size)
        else:
            rewards = reward_fn(batch_completions, expected_answers, group_size)

        # Normalize rewards for stability (mean=0, std=1)
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 1e-8:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards - reward_mean

        # Compute advantages using GRPO
        if use_stable_grpo:
            advantages = compute_grpo_advantages_stable(rewards_normalized, group_size)
        else:
            advantages = compute_grpo_advantages(
                rewards_normalized, group_size, beta=grpo_beta
            )

        # Compute loss and backward pass on training model
        loss, loss_metrics = train_group.forward_backward(
            batch_vllm_token_ids,
            batch_vllm_token_log_probs,
            batch_prompt_token_ids,
            advantages,
            num_rollout_batches,
        )
        total_loss += loss

        # Track metrics
        all_completions.extend(batch_completions[:2])  # Sample 2 from each batch
        all_rewards.append(reward_mean.item())
        all_advantages.append(advantages.mean().item())
        batch_metrics.append(loss_metrics)

    # Optimizer step (gradient clipping + weight update)
    train_group.optimizer_step()

    # Aggregate metrics across batches
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_advantage = sum(all_advantages) / len(all_advantages)

    # Use metrics from last batch for detailed stats
    final_metrics = batch_metrics[-1]

    # Return aggregated metrics
    metrics = {
        "loss": total_loss,
        "reward_mean": avg_reward,
        "reward_std": batch_metrics[-1].get("reward_std", 0.0),
        "advantage_mean": avg_advantage,
        "advantage_std": batch_metrics[-1].get("advantage_std", 0.0),
        "sample_completions": all_completions[:2],  # First 2 for inspection
        "num_rollout_batches": num_rollout_batches,
        "total_samples": len(prompt_texts) * group_size * num_rollout_batches,
        **final_metrics,  # Include final batch metrics
    }

    return metrics


def download_and_convert_model(
    model_name: str,
    cache_dir: str = "/mnt/local_storage/models",
    output_dir: str = "/mnt/local_storage/converted",
) -> tuple[str, str]:
    """
    Download model from HuggingFace and convert to TorchTitan format.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-1.7B")
        cache_dir: Directory to cache the downloaded model
        output_dir: Directory to save converted weights

    Returns:
        titan_checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to downloaded HuggingFace model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Download model from HuggingFace
    print(f"Downloading {model_name} from HuggingFace...")
    model_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.model"],
    )
    print(f"  Downloaded to: {model_path}")

    # Convert to TorchTitan format
    print("Converting weights to TorchTitan format...")
    titan_state = vllm_to_torchtitan(model_path)
    titan_checkpoint_path = os.path.join(output_dir, "qwen3_torchtitan.safetensors")
    save_file(titan_state, titan_checkpoint_path)
    print(f"  Saved TorchTitan weights to: {titan_checkpoint_path}")

    return output_dir, model_path


def load_model(checkpoint_path: str, model_path: str, use_vllm_compat: bool = True):
    """
    Load TorchTitan model from checkpoint.

    Args:
        checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model (for config)
        use_vllm_compat: If True, use vLLM-compatible model, else use standard model

    Returns:
        model: Loaded TorchTitan model
    """
    # Load HuggingFace config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create model args
    model_args = Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )

    # state_dict is in standard TorchTitan format (w1, w2, w3)
    state_dict = load_file(checkpoint_path)

    if use_vllm_compat:
        # Create and load model (using vLLM-compat for bitwise determinism)
        from torchtitan.experiments.deterministic_vllm_rl.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        # Convert to vLLM-compat format (merged gate_up_proj, down_proj)
        vllm_compat_state = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(vllm_compat_state, strict=False)
    else:
        # Use standard TorchTitan model
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)

    return model


def extract_numeric_answer(text: str) -> str | None:
    """
    Extract numeric answer from model completion.

    Looks for patterns like "#### 123" or final numbers in the text.

    Args:
        text: Completion text

    Returns:
        Extracted answer as string, or None if not found
    """
    # GSM8K uses #### to denote the final answer
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        # Remove commas from numbers
        return match.group(1).replace(",", "")

    # Fallback: look for last number in text
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def math_reward_function(
    completions: list[str],
    expected_answers: list[str],
    group_size: int = 4,
) -> torch.Tensor:
    """
    Reward function for math problems (e.g., GSM8K).

    Gives high reward for correct answers, low for incorrect.

    Args:
        completions: List of completion strings
        expected_answers: List of expected answers (one per prompt, repeated for group_size)
        group_size: Number of samples per prompt

    Returns:
        rewards: [batch]
    """
    rewards = []

    for idx, completion in enumerate(completions):
        # Map completion index to prompt index
        prompt_idx = idx // group_size
        expected = expected_answers[prompt_idx].strip().lower()

        # Extract answer from completion
        predicted = extract_numeric_answer(completion)

        if predicted is None:
            # No valid answer found
            reward = 0.0
        elif predicted.lower() == expected:
            # Correct answer
            reward = 1.0
        else:
            # Wrong answer
            reward = 0.0

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)


def load_gsm8k_dataset(split: str = "train", num_samples: int = 100):
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ("train" or "test")
        num_samples: Number of samples to load

    Returns:
        prompts: List of problem prompts
        answers: List of expected answers (numeric strings)
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=split)

        prompts = []
        answers = []

        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract the final numeric answer from the answer field
            # GSM8K answers are like "some explanation\n#### 42"
            answer_num = extract_numeric_answer(answer)
            if answer_num is None:
                continue

            # Format prompt for the model
            prompt = f"Question: {question}\nAnswer:"

            prompts.append(prompt)
            answers.append(answer_num)

        return prompts, answers

    except ImportError:
        print("⚠ datasets library not installed. Install with: pip install datasets")
        return None, None
    except Exception as e:
        print(f"⚠ Failed to load GSM8K dataset: {e}")
        return None, None


def trivial_reward_function(
    completions: list[str],
    tokenizer=None,
    expected_answers: list[str] | None = None,
    group_size: int = 4,
) -> torch.Tensor:
    """
    Reward function based on correctness and lowercase preference.

    Penalizes non-English characters to keep output in English.
    Rewards correct answers to factual questions.
    Penalizes capital letters to encourage lowercase output.

    Args:
        completions: List of completion strings
        tokenizer: Tokenizer to count tokens
        expected_answers: List of expected answers (one per prompt, repeated for group_size)
        group_size: Number of samples per prompt

    Returns:
        rewards: [batch]
    """
    batch_size = len(completions)
    rewards = []

    for idx, completion in enumerate(completions):
        # Start with base reward of 1.0
        reward = 1.0

        total_chars = len(completion)
        if total_chars == 0:
            rewards.append(0.0)
            continue

        # Penalty for non-English characters (keep it in English)
        # Count non-ASCII characters
        non_ascii_count = sum(1 for c in completion if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / total_chars
        # Strong penalty if >10% non-ASCII
        if non_ascii_ratio > 0.1:
            reward *= 0.1  # 10x penalty

        # Penalty for capital letters (encourage lowercase)
        uppercase_count = sum(1 for c in completion if c.isupper())
        uppercase_ratio = uppercase_count / total_chars
        # Apply penalty proportional to uppercase ratio
        # 0% uppercase = no penalty (1.0x)
        # 100% uppercase = strong penalty (0.1x)
        # Linear interpolation: penalty = 1.0 - 0.9 * uppercase_ratio
        uppercase_penalty = 1.0 - 0.9 * uppercase_ratio
        reward *= uppercase_penalty

        # Bonus for correct answers
        if expected_answers is not None:
            # Map completion index to prompt index
            prompt_idx = idx // group_size
            expected_answer = expected_answers[prompt_idx].lower()
            completion_lower = completion.lower()

            # Check if answer is in completion
            if expected_answer in completion_lower:
                reward *= 2.0  # 2x bonus for correct answer
            else:
                reward *= 0.5  # Penalty for wrong answer

        rewards.append(reward)

    rewards = torch.tensor(rewards, dtype=torch.float32)

    return rewards


def compute_grpo_advantages(
    rewards: torch.Tensor, group_size: int = 4, beta: float = 0.1
) -> torch.Tensor:
    """
    Compute advantages using GRPO-style exponential weighting.

    GRPO uses exponential advantages within groups which can be numerically
    unstable without bitwise determinism. Small differences in reward computation
    can lead to drastically different exp(reward/beta) values.

    This implementation uses the proper GRPO formulation:
    advantage_i = exp(reward_i / beta) / Z - 1
    where Z = mean(exp(reward_j / beta)) for j in group

    Args:
        rewards: [batch]
        group_size: Number of samples per prompt (batch must be divisible by this)
        beta: Temperature parameter for exponential weighting (lower = more unstable)

    Returns:
        advantages: [batch]
    """
    batch_size = rewards.shape[0]
    assert (
        batch_size % group_size == 0
    ), f"Batch size {batch_size} must be divisible by group_size {group_size}"

    num_groups = batch_size // group_size
    rewards_grouped = rewards.view(num_groups, group_size)

    # GRPO exponential advantages: exp(reward / beta)
    # This is numerically unstable and will explode without bitwise invariance!
    exp_rewards = torch.exp(rewards_grouped / beta)

    # Normalize by group mean (this is where instability shows up)
    group_mean_exp = exp_rewards.mean(dim=1, keepdim=True)

    # Advantage = normalized_exp - 1
    advantages_grouped = exp_rewards / group_mean_exp - 1.0

    # Flatten back
    advantages = advantages_grouped.view(-1)

    return advantages


def compute_grpo_advantages_stable(
    rewards: torch.Tensor, group_size: int = 4
) -> torch.Tensor:
    """
    Compute advantages using simple mean-centering (stable fallback).

    This is a simplified version that just uses mean-centering within groups.
    Use this if you want stable training without bitwise invariance.

    Args:
        rewards: [batch]
        group_size: Number of samples per prompt (batch must be divisible by this)

    Returns:
        advantages: [batch]
    """
    batch_size = rewards.shape[0]
    assert (
        batch_size % group_size == 0
    ), f"Batch size {batch_size} must be divisible by group_size {group_size}"

    num_groups = batch_size // group_size
    rewards_grouped = rewards.view(num_groups, group_size)

    # Compute advantages: reward - group_mean
    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    advantages_grouped = rewards_grouped - group_means

    # Flatten back
    advantages = advantages_grouped.view(-1)

    return advantages


def policy_gradient_loss(
    log_probs: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """
    Compute policy gradient loss.

    L = -E[log π(a|s) * A(s,a)]

    Args:
        log_probs: [batch, seq_len] - Log probs of generated tokens
        advantages: [batch] - Advantages for each sample

    Returns:
        loss: scalar
    """
    # Sum log probs across sequence for each sample
    total_log_probs = log_probs.sum(dim=1)  # [batch]

    # Policy gradient: -log_prob * advantage
    pg_loss = -(total_log_probs * advantages).mean()

    return pg_loss


def compute_policy_gradient_loss_vllm(
    model: torch.nn.Module,
    vllm_token_ids: list[list[int]],
    vllm_token_log_probs: list[list[float]],
    prompt_token_ids: list[list[int]],
    advantages: torch.Tensor,
    kl_coef: float = 0.1,
    ppo_clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO policy gradient loss by re-evaluating completions under current policy.

    Args:
        model: Current policy model
        vllm_token_ids: Generated token IDs for each completion
        vllm_token_log_probs: Per-token log probs from vLLM (reference)
        prompt_token_ids: Prompt token IDs for each completion
        advantages: [batch] - Advantages for each sample
        kl_coef: KL divergence penalty coefficient
        ppo_clip_eps: PPO clipping epsilon
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Total loss (PG + entropy + KL)
        metrics: Training metrics dict (includes per-token logprob deltas)
    """
    device = next(model.parameters()).device
    advantages = advantages.to(device)

    # Compute reference log probs from per-token values
    # Use PyTorch's sum() to match the reduction order used for total_log_probs
    # This ensures exactly zero KL divergence with batch invariance
    ref_log_probs = torch.stack(
        [
            torch.tensor(lps, dtype=torch.float32, device=device).sum()
            for lps in vllm_token_log_probs
        ]
    )

    # Compute log probs under current policy (WITH GRADIENTS)
    batch_token_log_probs = []
    batch_total_log_probs = []

    # Track per-token differences for the first sample
    first_sample_deltas = []

    for idx, (prompt_toks, gen_toks, vllm_toks_lp) in enumerate(
        zip(prompt_token_ids, vllm_token_ids, vllm_token_log_probs)
    ):
        # Concatenate prompt + generated tokens
        full_sequence = prompt_toks + gen_toks
        full_tensor = torch.tensor(
            full_sequence, dtype=torch.long, device=device
        ).unsqueeze(0)

        # Forward pass
        logits = model(full_tensor)
        if isinstance(logits, torch.distributed.tensor.DTensor):
            logits = logits.full_tensor()

        # Use F.log_softmax which is overridden by batch_invariant mode for determinism
        # Convert to float32 to match vLLM's sampler behavior (use .to() to preserve gradients)
        log_probs = F.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
        target_tokens = full_tensor[:, 1:]

        # Extract log probs for generated tokens only
        prompt_len = len(prompt_toks)
        gen_start_idx = prompt_len - 1
        gen_end_idx = gen_start_idx + len(gen_toks)

        gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
        gen_token_ids = target_tokens[0, gen_start_idx:gen_end_idx]
        token_lps = gen_token_logprobs.gather(1, gen_token_ids.unsqueeze(-1)).squeeze(
            -1
        )

        batch_token_log_probs.append(token_lps)
        batch_total_log_probs.append(token_lps.sum())

        # For the first sample, store raw tensors for bitwise comparison
        if idx == 0:
            # Keep bfloat16 tensors for bitwise comparison
            titan_lps_bf16 = token_lps.detach().cpu()  # Keep as bfloat16
            titan_lps_f32 = (
                token_lps.detach().cpu().float()
            )  # Convert to float32 for display

            for token_id, vllm_lp, titan_lp_bf16, titan_lp_f32 in zip(
                gen_toks, vllm_toks_lp, titan_lps_bf16, titan_lps_f32
            ):
                first_sample_deltas.append(
                    {
                        "token_id": token_id,
                        "vllm_logprob": vllm_lp,
                        "titan_logprob_bf16": titan_lp_bf16,
                        "titan_logprob_f32": titan_lp_f32.item(),
                    }
                )

    total_log_probs = torch.stack(batch_total_log_probs)

    # Verify bitwise determinism between vLLM and TorchTitan
    if first_sample_deltas:
        vllm_lps_f32 = torch.tensor(
            [d["vllm_logprob"] for d in first_sample_deltas], dtype=torch.float32
        )
        titan_lps_f32 = torch.tensor(
            [d["titan_logprob_f32"] for d in first_sample_deltas], dtype=torch.float32
        )

        bitwise_identical = torch.equal(vllm_lps_f32, titan_lps_f32)

        if bitwise_identical:
            print(
                f"  ✓ vLLM-TorchTitan bitwise determinism verified: {len(first_sample_deltas)} tokens match exactly"
            )
        else:
            num_different = (vllm_lps_f32 != titan_lps_f32).sum().item()
            deltas = (vllm_lps_f32 - titan_lps_f32).abs()
            max_delta = deltas.max().item()
            avg_delta = deltas.mean().item()
            print(
                f"  ⚠ vLLM-TorchTitan logprobs differ: {num_different}/{len(first_sample_deltas)} tokens"
            )
            print(f"    Max delta: {max_delta:.6e}, Avg delta: {avg_delta:.6e}")
            print(
                f"    vLLM logprobs:     {[f'{lp:.10f}' for lp in vllm_lps_f32[:5].tolist()]}"
            )
            print(
                f"    TorchTitan logprobs: {[f'{lp:.10f}' for lp in titan_lps_f32[:5].tolist()]}"
            )

    # PPO clipped objective
    log_ratio = total_log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)
    unclipped_loss = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)
    clipped_loss = clipped_ratio * advantages
    pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    # Entropy bonus
    all_token_log_probs = torch.cat(batch_token_log_probs)
    entropy = -all_token_log_probs.mean()
    entropy_bonus = -entropy_coef * entropy

    # KL divergence penalty
    kl_div = (ratio - 1 - log_ratio).mean()

    # Total loss
    total_loss = pg_loss + entropy_bonus + kl_coef * kl_div

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "kl_div": kl_div.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
        .float()
        .mean()
        .item(),
        "per_token_deltas": first_sample_deltas,  # Per-token logprob differences for first sample
    }

    return total_loss, metrics


def _check_if_batch_invariant_enabled(use_stable_grpo):
    # Check if batch invariance is enabled
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

    use_vllm_compat = vllm_is_batch_invariant()

    if use_vllm_compat:
        print("✓ Batch invariance detected - using vLLM-compatible model")
        # Add backward pass support to vLLM's batch_invariant mode
        print("  Adding gradient support to vLLM's batch_invariant mode...")
        from torchtitan.experiments.deterministic_vllm_rl.batch_invariant_backward import (
            enable_batch_invariant_backward_mode,
        )

        enable_batch_invariant_backward_mode()
    else:
        print("⚠ Batch invariance NOT detected - using standard model")
        if not use_stable_grpo:
            print(
                "  WARNING: Exponential GRPO may be unstable without bitwise invariance!"
            )


@ray.remote(num_cpus=4)
def main():
    """
    Simple RL training loop using vLLM for fast rollouts.

    Supports both single-GPU and distributed training with torchtitan parallelisms.
    To enable distributed training, set tp/fsdp/ddp > 1 in the config below.

    Parallelism options:
    - tp (Tensor Parallel): Splits model layers across GPUs (reduces per-GPU memory)
    - fsdp (FSDP): Shards parameters/gradients/optimizer (most memory efficient)
    - ddp (DDP): Replicates full model (higher throughput if model fits in memory)
    - Total training GPUs = tp × fsdp × ddp
    """

    # ========== Config ==========
    model_name = "Qwen/Qwen3-1.7B"  # HuggingFace model name
    cache_dir = "/mnt/local_storage/models"
    output_dir = "/mnt/local_storage/converted"

    # Parallelism config (tweak these for distributed training)
    tp = 1  # Tensor parallel degree (e.g., 2 = split model across 2 GPUs)
    fsdp = 2  # FSDP degree (e.g., 2 = shard parameters across 2 GPUs)
    ddp = 1  # DDP degree (e.g., 2 = replicate model on 2 GPUs)
    vllm_tp_size = 2  # vLLM tensor parallel size (number of GPUs for vLLM)
    train_world_size = tp * fsdp * ddp

    # Training config
    group_size = 8  # Samples per prompt for GRPO (increased from 4)
    num_rollout_batches = 2  # Multiple rollout batches per update (NEW!)
    num_steps = 100
    learning_rate = 1e-5

    # GRPO config
    use_stable_grpo = (
        False  # Set to True for stable training, False to test bitwise invariance
    )
    grpo_beta = 0.1  # Lower = more unstable (will explode without bitwise invariance!)

    # Dataset config
    use_real_dataset = (
        True  # Set to True to use GSM8K dataset (requires: pip install datasets)
    )
    num_dataset_samples = 10  # Number of prompts from dataset

    _check_if_batch_invariant_enabled(use_stable_grpo)

    # Print parallelism config
    num_train_gpus = tp * fsdp * ddp
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Training GPUs: {num_train_gpus} (TP={tp}, FSDP={fsdp}, DDP={ddp})")
    print(f"vLLM GPUs: {vllm_tp_size}")
    print(f"Total GPUs: {num_train_gpus + vllm_tp_size}")
    print("=" * 80)

    # Download and convert model
    print("=" * 80)
    print(f"Setting up model: {model_name}")
    print("=" * 80)
    titan_checkpoint_path, model_path = download_and_convert_model(
        model_name, cache_dir, output_dir
    )

    # Check if batch invariance is enabled
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

    use_vllm_compat = vllm_is_batch_invariant()

    # Initialize persistent vLLM engine for rollouts
    print(f"\nInitializing vLLM engine for rollouts (TP size: {vllm_tp_size})...")
    vllm_engine = VLLMRolloutEngine.options(num_gpus=0).remote(
        model_path, tp_size=vllm_tp_size
    )

    # Create TrainGroup for training (now an orchestrator)
    print("\nInitializing TrainGroup...")
    train_group = TrainGroup(
        titan_checkpoint_path=model_path,
        model_path=model_path,
        use_vllm_compat=use_vllm_compat,
        learning_rate=learning_rate,
        tp=tp,
        fsdp=fsdp,
        ddp=ddp,
    )

    # Set up weight update groups (NCCL communication)
    # Only rank 0 training worker + all vLLM workers participate
    print("\nSetting up weight update groups...")
    from vllm.utils.network_utils import get_ip, get_open_port

    master_address = get_ip()
    master_port = get_open_port()

    # world_size = 1 training worker (rank 0) + vLLM workers
    weight_update_world_size = 1 + vllm_tp_size
    vllm_rank_offset = 1  # vLLM workers start at rank 1

    # Initialize weight update group on vLLM workers
    print(
        f"Initializing vLLM workers with master_address={master_address}, master_port={master_port}"
    )
    handle = vllm_engine.init_weight_update_group.remote(
        master_address, master_port, vllm_rank_offset, weight_update_world_size
    )

    # Initialize weight update group on training worker rank 0
    train_group.init_weight_update_group(
        master_address, master_port, weight_update_world_size
    )

    # Wait for vLLM workers to join
    ray.get(handle)
    print("✓ Weight update groups initialized")

    # Load tokenizer for reward computation
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Tokenizer loaded")

    # Load dataset
    print("\n" + "=" * 80)
    print("Dataset Configuration")
    print("=" * 80)

    if use_real_dataset:
        print(f"Attempting to load GSM8K dataset ({num_dataset_samples} samples)...")
        prompt_texts, expected_answers = load_gsm8k_dataset(
            split="train", num_samples=num_dataset_samples
        )

        if prompt_texts is None or len(prompt_texts) == 0:
            print("⚠ Failed to load dataset, falling back to default prompts")
            use_real_dataset = False

    if not use_real_dataset:
        # Fallback: simple prompts with verifiable answers
        print("Using default prompts (factual questions)")
        prompts_with_answers = [
            ("The capital of France is", "paris"),
            ("What is 7 times 8?", "56"),
            ("The first president of the United States was", "washington"),
            ("The chemical symbol for water is", "h2o"),
            ("The largest planet in our solar system is", "jupiter"),
        ]
        prompt_texts = [p[0] for p in prompts_with_answers]
        expected_answers = [p[1] for p in prompts_with_answers]

    # Select reward function
    reward_fn = math_reward_function if use_real_dataset else trivial_reward_function

    print(f"Loaded {len(prompt_texts)} prompts")
    print(f"Reward function: {reward_fn.__name__}")
    print(f"First prompt: {prompt_texts[0][:80]}...")

    # TensorBoard writer
    writer = SummaryWriter("/mnt/local_storage/output/rl_training")
    print("\n" + "=" * 80)
    print("TensorBoard logging enabled at: /mnt/local_storage/output/rl_training")
    print("=" * 80)

    # Training loop
    print(f"\nStarting RL training for {num_steps} steps...")
    print(f"  Prompts: {len(prompt_texts)}")
    print(f"  Samples per prompt: {group_size}")
    print(f"  Rollout batches per update: {num_rollout_batches}")
    print(
        f"  Total samples per update: {len(prompt_texts) * group_size * num_rollout_batches}"
    )
    print(
        f"  GRPO mode: {'Stable (mean-centering)' if use_stable_grpo else f'Exponential (beta={grpo_beta})'}"
    )
    print(f"  vLLM tensor parallel size: {vllm_tp_size}")
    print("=" * 80)
    from tqdm import tqdm

    for step in tqdm(range(num_steps), desc="Training"):
        metrics = rl_update_step(
            train_group,
            vllm_engine,
            tokenizer,
            prompt_texts,
            expected_answers=expected_answers,
            group_size=group_size,
            max_new_tokens=20 if not use_real_dataset else 100,
            temperature=1.0,
            num_rollout_batches=num_rollout_batches,
            reward_fn=reward_fn,
            grpo_beta=grpo_beta,
            use_stable_grpo=use_stable_grpo,
        )

        # Log to TensorBoard
        writer.add_scalar("rl/loss", metrics["loss"], step)
        writer.add_scalar("rl/pg_loss", metrics["pg_loss"], step)
        writer.add_scalar("rl/kl_div", metrics["kl_div"], step)
        writer.add_scalar("rl/entropy", metrics["entropy"], step)
        writer.add_scalar("rl/ratio_mean", metrics["ratio_mean"], step)
        writer.add_scalar("rl/ratio_clipped_frac", metrics["ratio_clipped_frac"], step)
        writer.add_scalar("rl/reward_mean", metrics["reward_mean"], step)
        writer.add_scalar("rl/reward_std", metrics.get("reward_std", 0.0), step)
        writer.add_scalar("rl/advantage_mean", metrics["advantage_mean"], step)
        writer.add_scalar("rl/advantage_std", metrics.get("advantage_std", 0.0), step)
        writer.add_scalar("rl/total_samples", metrics["total_samples"], step)

        # TODO: fix compute weight deltas
        # # Compute weight deltas from initial state
        # weight_deltas = train_group.compute_weight_deltas()

        # # Log weight deltas
        # for key, value in weight_deltas.items():
        #     writer.add_scalar(key, value, step)

        print(
            f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:+.3f} | "
            f"Samples: {metrics['total_samples']}"
        )
        print(f"  Sample: {metrics['sample_completions'][0][:80]}...")

        # Check for NaN/Inf (sign of instability)
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            print("\n" + "!" * 80)
            print("ERROR: Loss is NaN/Inf! Training diverged.")
            print(
                "This likely means the exponential GRPO is unstable without bitwise invariance."
            )
            print("Try setting use_stable_grpo=True or enabling batch invariance mode.")
            print("!" * 80)
            break

    print("\n" + "=" * 80)
    print("Training complete!")
    print(
        "View TensorBoard: tensorboard --logdir=/mnt/local_storage/output/rl_training"
    )
    print("=" * 80)

    # Cleanup
    writer.close()
    del vllm_engine


if __name__ == "__main__":
    # Initialize Ray
    print("\nInitializing Ray...")
    ray.init()

    ray.get(main.remote())
