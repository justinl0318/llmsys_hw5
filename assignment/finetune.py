"""Part C: DeepSpeed ZeRO and LoRA integration (30 points)."""

from re import L
import torch
from torch import nn
from assignment.runtime import LoRALinear


def build_config(world_size, micro_batch, accumulation, precision="fp32"):
    """Return a DeepSpeed ZeRO-2 configuration, with no CPU/NVMe offload.

    global batch = world_size * micro_batch * accumulation. Accept fp32/fp16;
    reject bf16 in this V100 assignment. Set gradient clipping to zero for
    comparison with the serial reference. The grader supplies torch.optim.SGD;
    allow this optimizer with ZeRO. No optimizer or scheduler definition needed.
    """
    # BEGIN_STUDENT
    if world_size < 1 or micro_batch < 1 or accumulation < 1:
        raise ValueError("world_size < 1 or microbatch < 1 or accumulation < 1")
    
    if precision != "fp32" and precision != "fp16":
        raise ValueError("precision must be fp32 or fp16")
    
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "train_batch_size": world_size * micro_batch * accumulation,
        "zero_optimization": {"stage": 2},
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 0,
        "fp16": {"enabled": precision == "fp16"},
    }
    # END_STUDENT


def apply_lora(model, target_names, rank=2, alpha=4):
    """Replace exactly the named nn.Linear modules with supplied LoRALinear.

    Validate all targets BEFORE mutating anything. Reject empty, duplicate,
    missing, or non-Linear targets and rank < 1 with ValueError. Freeze all base
    parameters; only lora_A and lora_B are trainable. Preserve device/dtype.
    Return the same model object; dotted nested names are supported.
    """
    # BEGIN_STUDENT
    if rank < 1:
        raise ValueError("rank < 1")
    
    target_names = list(target_names)
    if not target_names:
        raise ValueError("empty target_names")
    if len(set(target_names)) != len(target_names):
        raise ValueError("contain duplicate target names")
    
    # validate each target
    modules = dict(model.named_modules())
    for target_name in target_names:
        if target_name not in modules:
            raise ValueError(f"module {target_name} not found")
        if not isinstance(modules[target_name], nn.Linear):
            raise ValueError(f"module {target_name} is not Linear type")
        
    # freeze all base params
    for param in model.parameters():
        param.requires_grad_(False)
        
    # replace each target param with LoRA param
    for target_name in target_names:
        parent_name, sep, layer_name = target_name.rpartition(".") # split at the last "."
        parent = modules[parent_name]
        
        base_layer = modules[target_name]
        lora_layer = LoRALinear(base_layer, rank, alpha)
        setattr(parent, layer_name, lora_layer)
        
    return model
    # END_STUDENT


def save_adapter(model, path):
    """Save CPU copies of ONLY named lora_A/lora_B tensors to path.

    Format is a flat name -> tensor dict loadable with weights_only=True.
    The harness calls this on a normal/gathered model, not partitioned ZeRO weights.
    Reject a model without adapters. Parent directory already exists.
    """
    # BEGIN_STUDENT
    adapters = {}
    for name, param in model.named_parameters():
        if name.split(".")[-1] in ("lora_A", "lora_B"):
            adapters[name] = param.detach().cpu().clone()
            
    if not adapters:
        raise ValueError("empty LoRA adapters")
    
    torch.save(adapters, path)
    # END_STUDENT


def load_adapter(model, path):
    """Restore a saved adapter into a model with matching adapters.

    Reject missing/extra keys or mismatched shapes before any parameter changes.
    Preserve the model's existing base weights and parameter registration.
    """
    # BEGIN_STUDENT
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    
    adapters = {}
    for name, param in model.named_parameters():
        if name.split(".")[-1] in ("lora_A", "lora_B"):
            adapters[name] = param
            
    if loaded.keys() != adapters.keys():
        raise ValueError("adapter names don't match")
    
    for name, param in adapters.items():
        if not isinstance(loaded[name], torch.Tensor):
            raise ValueError(f"loaded {name} is not Tensor")
        if loaded[name].shape != param.shape:
            raise ValueError(f"loaded {name}'s tensor shape mismatch")
        
    with torch.no_grad():
        for name, param in adapters.items():
            param.copy_(loaded[name])
        
    # END_STUDENT


def train_microbatch(engine, inputs, targets, loss_fn):
    """Forward, engine.backward(loss), engine.step(); return detached local loss.

    DeepSpeed owns accumulation/gradient clearing/optimizer stepping. Do not
    divide loss or bypass the engine with ordinary loss.backward/optimizer.step.
    """
    # BEGIN_STUDENT
    outputs = engine(inputs)
    
    loss = loss_fn(outputs, targets)
    engine.backward(loss)
    engine.step()
    
    return loss.item()
    # END_STUDENT
