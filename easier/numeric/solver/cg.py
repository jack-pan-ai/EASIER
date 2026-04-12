# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Dict, Any

import torch
from torch import nn

import easier as esr


class Init(esr.Module):

    def __init__(self, A, b, x, M, r, z, p, bnorm) -> None:
        super().__init__()
        self.A = A
        self.x = x
        self.b = b
        self.M = M
        self.r = r
        self.z = z
        self.p = p
        self.bnorm = bnorm

    def forward(self):
        self.r[:] = self.b - self.A(self.x)
        self.z[:] = self.M(self.r)
        self.p[:] = self.z
        self.bnorm[:] = esr.norm(self.b).norm()


class Step(esr.Module):

    def __init__(self, A, x, r, z, p, rnorm, rzsum):
        super().__init__()
        self.A = A
        self.x = x
        self.p = p
        self.r = r
        self.z = z
        self.rnorm = rnorm
        self.rzsum = rzsum

    def forward(self):
        Ap = self.A(self.p)
        rzsum = esr.sum(self.r * self.z).sum()
        alpha = rzsum / esr.sum(self.p * Ap).sum()
        r1 = self.r - alpha * Ap
        self.x.add_(alpha * self.p)
        self.rzsum[:] = rzsum
        self.r[:] = r1
        self.rnorm[:] = esr.norm(r1).norm()


class Update(esr.Module):

    def __init__(self, x, M, r, z, p, rzsum):
        super().__init__()
        self.x = x
        self.M = M
        self.r = r
        self.z = z
        self.p = p
        self.rzsum = rzsum

    def forward(self):
        z1 = self.M(self.r)
        beta = esr.sum(self.r * z1).sum() / self.rzsum
        p1 = z1 + beta * self.p
        self.z[:] = z1
        self.p[:] = p1


class CG(esr.Module):
    """Conjugate Gradient"""

    def __init__(self,
                 A: nn.Module,
                 b: esr.Tensor,
                 x: esr.Tensor,
                 M: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.A = A
        self.b = b
        self.x = x
        self.M = M if M else lambda x: x

        self.r = esr.Tensor(esr.zeros_like(x), mode='partition')
        self.z = esr.Tensor(esr.zeros_like(x), mode='partition')
        self.p = esr.Tensor(esr.zeros_like(x), mode='partition')

        self.rzsum = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')
        self.rnorm = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')
        self.bnorm = esr.Tensor(
            torch.tensor([0.0], dtype=x.dtype, device=x.device),
            mode='replicate')

        self.init = Init(A, b, x, self.M, self.r, self.z, self.p, self.bnorm)
        self.step = Step(A, x, self.r, self.z, self.p, self.rnorm, self.rzsum)
        self.update = Update(x, self.M, self.r, self.z, self.p, self.rzsum)

    def solve(self,
              rtol: float = 1e-5,
              atol: Optional[float] = None,
              maxiter: Optional[int] = None,
              debug_iter: Optional[int] = None,
              profile: Optional[bool] = False
              ) -> Dict[str, Any]:
        name = self.__class__.__name__
        self.init()
        rtol *= self.bnorm
        tol = max(rtol, atol) if atol else rtol

        iters = 0
        while True:
            self.step()

            if debug_iter is not None and iters % debug_iter == 0:
                esr.logger.info(
                    f"{name} residual {float(self.rnorm)}"
                    f" at the {iters}-th iteration")

            if (not torch.isnan(self.rnorm) and self.rnorm <= tol) or \
            (maxiter is not None and iters >= maxiter):
                break
            iters += 1

            self.update()

        esr.logger.info(
            f"{name} solver completed with residual {float(self.rnorm)}" +
            f" at the {iters}-th iteration")
        return {'residual': float(self.rnorm), 'iters': iters}
    
    def solve_profile(self,
              maxiter: Optional[int] = None,
              ) -> Dict[str, Any]:
        self.init()

        for _ in range(maxiter):
            self.step()
            self.update()
    
    def c_profile_timing_cg(self):
        import cProfile
        import time
        import csv
        import os
        self.init()

        prof = cProfile.Profile()
        prof.enable()
        start_time = time.time()
        for _ in range(5):
            self.step()
            self.update()
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        avg_time_ms = avg_time * 1000
        print(f"eqn() execution time: {avg_time_ms:.6f} ms")
        prof.disable()
        prof.dump_stats(f'res/profile-poisson.prof')
        # Save timing result to a CSV file with device, backend, and dt in the filename
        csv_dir = "res"
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = os.path.join(csv_dir, f"timing_poisson_cpu_torch.csv")
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["num_times", "avg_ms_per_iter"])
            writer.writerow([5, f"{avg_time_ms:.6f}"])

    def torch_profile(self):
        for _ in range(1):
            self.step()
            self.update()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                # torch.profiler.ProfilerActivity.CUDA,
            ],
            # record_shapes=True,
            with_stack=True,
            with_modules=True,
            # with_flops=False,
        ) as prof:
            self.step()
            self.update()
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        prof.export_chrome_trace(f"torchprof-cg-cpu.json")
