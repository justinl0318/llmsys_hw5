"""Part B: microbatch pipeline parallelism (35 points)."""

import torch
from torch import nn
from assignment.runtime import StageWorkers


def clock_cycles(microbatches, stages):
    """Yield complete wavefronts of (microbatch, stage) pairs.

    Pair (i,j) runs at cycle i+j. Each cycle is ordered by increasing stage.
    Zero microbatches yields nothing. Reject negative microbatches or stages < 1.
    """
    # BEGIN_STUDENT
    if microbatches < 0 or stages < 1:
        raise ValueError("microbatches < 0 or stages < 1")
    
    if microbatches == 0:
        return
    
    # num of clock cycles = microbatches + stages - 1
    for cycle in range(microbatches + stages - 1):
        # wave_i has microbatch + stage = i
        wave = []
        for stage in range(stages):
            microbatch = cycle - stage
            if microbatch >= 0 and microbatch < microbatches:
                wave.append((microbatch, stage))
    
        yield wave
            
    # END_STUDENT


class Pipe(nn.Module):
    """Provided stages are nn.Modules already placed on their assigned devices.

    Inputs have a nonempty leading batch dimension. split_size means examples
    per microbatch, not the number of microbatches. Stages return tensors.
    The supplied workers handle thread autograd mode and CUDA completion.
    Use submit(stage, callable) for the whole wave before taking any results
    with receive(stage). Keep tensors attached to autograd through transfers,
    and do every cross-device transfer inside a submitted callable, not in the
    calling thread; otherwise the stages serialize and lose the speedup.
    Return outputs in original example order on the last device.
    """

    def __init__(self, stages, devices, split_size):
        super().__init__()
        if not stages or len(stages) != len(devices) or split_size < 1:
            raise ValueError("invalid pipeline configuration")
        self.stages = nn.ModuleList(stages)
        self.devices = [torch.device(d) for d in devices]
        self.split_size = split_size
        self.workers = StageWorkers(self.devices)

    def forward(self, inputs):
        # BEGIN_STUDENT
        microbatches = list(torch.split(inputs, self.split_size, dim=0))
        for wave in clock_cycles(len(microbatches), len(self.stages)):
            # run each microbatch on the stage's worker thread
            for microbatch_idx, stage in wave:
                microbatch = microbatches[microbatch_idx]
                module = self.stages[stage]
                device = self.devices[stage]
                
                def run_curr_stage(input=microbatch, module=module, device=device):
                    return module(input.to(device))
                
                self.workers.submit(stage, run_curr_stage)
                
            # collect each microbatch output
            for microbatch_idx, stage in wave:
                microbatches[microbatch_idx] = self.workers.receive(stage)
             
        # last stage produces every microbatch's final output on its GPU
        # => no extra transfer needed
        return torch.cat(microbatches, dim=0) # concat back to a full batch
        # END_STUDENT

    def close(self):
        self.workers.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
