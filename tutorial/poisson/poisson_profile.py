# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import time
import csv
import torch

import matplotlib.pyplot as plt
from matplotlib import tri

import easier as esr
from easier.numeric import Linsys
from easier.numeric.solver import CG, GMRES


def run_profile_timing(sol, args):
    # warmup
    num_warmup = 5 if args.device == 'cpu' else 50
    # compilation
    print("Compiling...")
    sol.solve_profile(maxiter=1)
    print(f"Warming up {num_warmup} times")
    _ = sol.solve_profile(maxiter=num_warmup)
    # profile runs
    num_times = 20 if args.device == 'cpu' else 100
    print(f"Running {num_times} times")
    if args.device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    # maxiter
    _ = sol.solve_profile(maxiter=num_times)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(
        f"Time to run solve() {num_times} times: {elapsed:.4f} seconds, "
        f"{elapsed/num_times*1000:.4f} ms per iteration"
    )
    # Save timing result to a CSV file with device and backend in the path
    if args.output:
        csv_filename = f"{args.output}/timing_poisson_{args.device}_{args.backend}.csv"
    else:
        csv_filename = f"timing_poisson_{args.device}_{args.backend}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    write_header = not os.path.exists(csv_filename)
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["num_times", "seconds", "ms_per_iteration"])
        writer.writerow(
            [num_times, f"{elapsed:.6f}", f"{elapsed/num_times*1000:.4f}"]
        )


class Poisson(esr.Module):
    def __init__(self, mesh: str, poisson: str, device='cpu', x=None) -> None:
        super().__init__()

        # src (torch.LongTensor): src cell indices, with shape `(ne,)`
        self.src = esr.hdf5(mesh, 'src', dtype=torch.long)
        # dst (torch.LongTensor): dst cell indices, with shape `(ne,)`
        self.dst = esr.hdf5(mesh, 'dst', dtype=torch.long)

        cells = esr.hdf5(mesh, 'cells', dtype=torch.long)
        self.nc = cells.shape[0]

        self.reducer = esr.Reducer(self.src, self.nc)
        self.selector = esr.Selector(self.dst)

        self.x = esr.Tensor(
            esr.zeros((self.nc,), dtype=torch.double), mode='partition'
        ) if x is None else x
        # b: (nc,)
        self.b = esr.Tensor(
            esr.hdf5(poisson, 'b', dtype=torch.double), mode='partition')
        # Ac: (nc,)
        self.Ac = esr.Tensor(
            esr.hdf5(poisson, 'Ac', dtype=torch.double), mode='partition')
        # Af: (src.shape[0],)
        self.Af = esr.Tensor(
            esr.hdf5(poisson, 'Af', dtype=torch.double), mode='partition'
        )
        self.A = Linsys(self.Ac, self.Af, self.selector, self.reducer)

        self.rho = esr.Tensor(
            esr.hdf5(poisson, 'rho', dtype=torch.double), mode='partition')
        # centroid: (nc, 2)
        self.centroid = esr.Tensor(
            esr.hdf5(poisson, 'centroid', dtype=torch.double), mode='partition'
        )

        self.to(device)


if __name__ == '__main__':
    """
    Usage:

    torchrun --nproc_per_node=4 tutorial/poisson/poisson_main.py \
        --solver=cg --backend=cpu --comm_backend=gloo \
        ~/.easier/triangular_100.hdf5 ~/.easier/Poisson_100.hdf5
    torchrun tutorial/poisson/poisson_main.py \
        --solver=cg --backend=torch --comm_backend=gloo --debug_iter=10\
        ~/.easier/triangular_500.hdf5 ~/.easier/Poisson_500.hdf5
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["cg", "gmres"], default="cg"
    )
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--backend", type=str, choices=["none", "torch", "cpu", "cuda"],
        default='torch'
    )
    parser.add_argument(
        "--comm_backend", type=str, choices=["gloo", "nccl"],
        default='gloo'
    )
    parser.add_argument(
        "--threads", type=int,
        help="Torch intra-op thread count; also used for OMP/MKL/OPENBLAS if set",
        default=20,
    )
    parser.add_argument(
        "--interop_threads", type=int,
        help="Torch inter-op thread count",
        default=1,
    )
    parser.add_argument("--debug_iter", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=str)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("mesh", type=str)
    parser.add_argument("poisson", type=str)
    args = parser.parse_args()

    print("Compile Poisson:")
    print("mesh HDF5 file:   ", args.mesh)
    print("poisson HDF5 file:", args.poisson)

    # Configure thread counts if provided so runs are repeatable.
    if args.threads:
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    if args.interop_threads:
        torch.set_num_interop_threads(args.interop_threads)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"[cfg] torch threads={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}, "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
        )

    esr.init(args.comm_backend)

    eqn = Poisson(args.mesh, args.poisson, args.device)

    if args.solver == 'cg':
        sol = CG(eqn.A, eqn.b, eqn.x)
    else:
        sol = GMRES(eqn.A, eqn.b, eqn.x)

    eqn, sol = esr.compile([eqn, sol], args.backend)

    if args.profile:
        run_profile_timing(sol, args)
    else:
        info = sol.solve(
            maxiter=args.maxiter,
            atol=args.atol,
            debug_iter=args.debug_iter
        )
        assert info["residual"] < args.atol

        if args.plot:
            rho = eqn.rho.collect().cpu().numpy()
            x_synced = eqn.x.collect().cpu().numpy()
            centroid = eqn.centroid.collect().cpu().numpy()

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                cells = tri.Triangulation(centroid[::1, 0], centroid[::1, 1])

                plt.figure(figsize=(6, 5))
                im = plt.tricontourf(cells, rho[::1], levels=50, cmap='jet')
                # plt.triplot(points, linewidth=0.1)
                plt.colorbar(im)
                plt.tight_layout()
                plt.savefig('rho.jpeg', dpi=300)
                plt.cla()

                plt.figure(figsize=(6, 5))
                im = plt.tricontourf(
                    cells, x_synced[::1], levels=50, cmap='jet'
                )
                # plt.triplot(points, linewidth=0.1)
                plt.colorbar(im)
                plt.tight_layout()
                plt.savefig('phi.jpeg', dpi=300)
                plt.cla()
