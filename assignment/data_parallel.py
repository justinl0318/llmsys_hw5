"""Part A: manual data parallelism (35 points)."""

import random
import torch.distributed as dist


def partition_indices(size, world_size, seed=1234):
    """Return world_size disjoint balanced lists covering range(size).

    Args:
        size: the # of training examples
        world_size: the # of GPUs
    
    Returns:
        a list of `world_size` list, indicating the indices of training examples each GPU should process
        
    Use a local random.Random(seed); do not modify global RNG state.
    Give the first size % world_size ranks one extra example. Allow empty ranks.
    Reject negative size or nonpositive world_size with ValueError.
    """
    # BEGIN_STUDENT
    if size < 0 or world_size < 1:
        raise ValueError("size is < 0 or world_size is < 1")
    
    indices = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    partitioned_size = size // world_size
    extra_ranks = size % world_size # the ranks who should get 1 extra
    partitions = []
    start_idx = 0
    # ex: size = 11, world_size = 3
    # => GPU0 gets 4, GPU1 gets 4, GPU2 gets 3
    for rank in range(world_size):
        rank_size = partitioned_size
        if rank < extra_ranks:
            rank_size += 1
            
        partitions.append(indices[start_idx:start_idx + rank_size])
        start_idx += rank_size
        
    return partitions
    # END_STUDENT


def average_gradients(model):
    """In-place arithmetic mean across ranks using dist.all_reduce.

    An initialized process group is required. Parameters with grad=None on ALL
    ranks are unused and must remain None. All ranks have the same gradient
    presence pattern and parameter order. The harness uses equal local batch
    sizes and equal normalization, so the arithmetic mean is the correct rule.
    Do not use DistributedDataParallel or recreate/detach model parameters.
    """
    # BEGIN_STUDENT
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is None: # skip parameters that were not used in the forward pass
            continue
        
        # can see this blogpost on how all_reduce is performed under the hood:
        # https://www.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model-synchronization-in-multi-gpu-training/
        # Internally, it is performed as ring-reduce with reduce-scatter + all-gather
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)
    # END_STUDENT


def train_step(model, optimizer, inputs, targets, loss_fn):
    """Clear gradients, forward, backward, average gradients, then update once.

    Return the detached scalar local loss as a Python float.
    """
    # BEGIN_STUDENT
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    
    average_gradients(model)
    optimizer.step()
    return loss.item()
    # END_STUDENT
