#! /bin/bash
# if jit compile is too slow, comment this one;
# this will save the cache and reuse it for next run.
export EASIER_DISABLE_JIT_HASH=1

export THREADS=20
export OMP_PLACES="{0:20}"
export OMP_PROC_BIND=close
# export OMP_DISPLAY_ENV=verbose
# export OMP_DISPLAY_AFFINITY=TRUE
export LD_PRELOAD=$CONDA_PREFIX/lib/libtcmalloc.so.4:$LD_PRELOAD

# Problem sizes (same as shallow water example)
N_CPU=(1000 2000 3000 4000 5000)
N_GPU=(1000 2000 3000 4000 5000)

mkdir -p res

# ----------------------------------------------------------------------
# Data generation: triangular meshes
# ----------------------------------------------------------------------
for n in ${N_CPU[@]}
do
    echo "Generating data for CPU case with n=${n}"
    if [ ! -f ~/.easier/triangular_${n}.hdf5 ]; then
        echo "  Creating triangular mesh for n=${n}"
        python tutorial/create_triangular_mesh.py ${n} ~/.easier/
    else
        echo "  Triangular mesh for n=${n} already exists, skipping mesh generation."
    fi
done

for n in ${N_GPU[@]}
do
    echo "Generating data for GPU case with n=${n}"
    if [ ! -f ~/.easier/triangular_${n}.hdf5 ]; then
        echo "  Creating triangular mesh for n=${n}"
        python tutorial/create_triangular_mesh.py ${n} ~/.easier/
    else
        echo "  Triangular mesh for n=${n} already exists, skipping mesh generation."
    fi
done

# ----------------------------------------------------------------------
# Assemble Poisson data
# ----------------------------------------------------------------------
for n in ${N_CPU[@]}
do
    echo "Checking/Creating Poisson data for CPU case with n=${n}"
    if [ ! -f ~/.easier/Poisson_${n}.hdf5 ]; then
        echo "  Poisson data missing for n=${n}, generating."
        torchrun tutorial/poisson/assemble_poisson.py \
            --device=cpu --comm_backend gloo \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5
    else
        echo "  Poisson data for n=${n} already exists, skipping Poisson data generation."
    fi
done

for n in ${N_GPU[@]}
do
    echo "Checking/Creating Poisson data for GPU case with n=${n}"
    if [ ! -f ~/.easier/Poisson_${n}.hdf5 ]; then
        echo "  Poisson data missing for n=${n}, generating."
        torchrun tutorial/poisson/assemble_poisson.py \
            --device=cuda --comm_backend nccl \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5
    else
        echo "  Poisson data for n=${n} already exists, skipping Poisson data generation."
    fi
done

# ----------------------------------------------------------------------
# Main simulation / profiling (CPU)
# ----------------------------------------------------------------------
for n in ${N_CPU[@]}
do
    echo "Running main Poisson solve on CPU for n=${n}"

    # # CPU Torch backend
    echo "  [CPU][torch backend] Profiling Poisson solve"
    torchrun --nproc_per_node=1 tutorial/poisson/poisson_profile.py \
        --solver=cg --profile=True --backend=torch \
        --maxiter=100 --atol=1e-10 --debug_iter=10 \
        --device=cpu --comm_backend=gloo --output=res/ \
        --threads=${THREADS} --interop_threads=1 \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5 \

    # CPU Easier JIT backend
    echo "  [CPU][jit backend] Profiling Poisson solve"
    torchrun --nproc_per_node=1 tutorial/poisson/poisson_profile.py \
        --solver=cg --profile=True --backend=cpu \
        --maxiter=100 --atol=1e-10 --debug_iter=10 \
        --device=cpu --comm_backend=gloo --output=res/ \
        --threads=${THREADS} --interop_threads=1 \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5
done

# ----------------------------------------------------------------------
# Main simulation / profiling (GPU)
# Uncomment this block if you want to profile GPU as well.
# ----------------------------------------------------------------------
for n in ${N_GPU[@]}
do
    echo "Running main Poisson solve on GPU for n=${n}"

    # CUDA Torch backend
    echo "  [CUDA][torch backend] Profiling Poisson solve"
    torchrun tutorial/poisson/poisson_profile.py \
        --solver=cg --profile=True --backend=torch \
        --maxiter=100 --atol=1e-10 --debug_iter=10 \
        --device=cuda --comm_backend=nccl --output=res/ \
        --threads=${THREADS} --interop_threads=1 \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5

    # CUDA Easier JIT backend
    echo "  [CUDA][jit backend] Profiling Poisson solve"
    torchrun tutorial/poisson/poisson_profile.py \
        --solver=cg --profile=True --backend=cuda \
        --maxiter=100 --atol=1e-10 --debug_iter=10 \
        --device=cuda --comm_backend=nccl --output=res/ \
        --threads=${THREADS} --interop_threads=1 \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/Poisson_${n}.hdf5
done

