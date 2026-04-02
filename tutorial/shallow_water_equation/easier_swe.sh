#! /bin/bash
# if jit compile is too slow, comment this one;
# this will save the cache and reuse it for next run.
# export EASIER_DISABLE_JIT_HASH=1

export OMP_NUM_THREADS=32
export OMP_PLACES="{0:32}"
export OMP_PROC_BIND=close
export OMP_DISPLAY_ENV=verbose
export OMP_DISPLAY_AFFINITY=TRUE

N_CPU=(500 1000 1500 2000 2500 3000)
N_GPU=(500 1000 1500 2000 2500 3000)

# N_CPU=(1000 2000 3000 4000 5000)
# N_GPU=(2000)
# N_CPU=(2000)
# N_GPU=(500)

mkdir res

# data genereation
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

# create SW data
for n in ${N_CPU[@]}
do
    echo "Checking/Creating SW data for CPU case with n=${n}"
    if [ ! -f ~/.easier/SW_${n}.hdf5 ]; then
        echo "  Triangular mesh or SW data missing for n=${n}, re-generating."
        torchrun tutorial/shallow_water_equation/assemble_shallow_water.py \
            --device=cpu --backend torch --comm_backend gloo \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5
    else
        echo "  SW data for n=${n} already exists, skipping SW data generation."
    fi
done

for n in ${N_GPU[@]}
do
    echo "Checking/Creating SW data for GPU case with n=${n}"
    if [ ! -f ~/.easier/SW_${n}.hdf5 ]; then
        echo "  Triangular mesh or SW data missing for n=${n}, re-generating."
        torchrun tutorial/shallow_water_equation/assemble_shallow_water.py \
            --device=cuda --backend torch --comm_backend nccl \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5
    else
        echo "  SW data for n=${n} already exists, skipping SW data generation."
    fi
done

# main simulation
for n in ${N_CPU[@]}
do
    echo "Running main simulation on CPU for n=${n}"
    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cpu torch
    echo "  [CPU][torch backend] Running simulation with dt=${delta_t}"
    python tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=torch \
        --device=cpu --comm_backend=gloo --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
    # cpu jit
    echo "  [CPU][jit backend] Running simulation with dt=${delta_t}"
    torchrun --nproc_per_node=1 tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=cpu \
        --device=cpu --comm_backend=gloo --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
done


for n in ${N_GPU[@]}
do
    echo "Running main simulation on GPU for n=${n}"
    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cuda torch
    echo "  [CUDA][torch backend] Running simulation with dt=${delta_t}"
    torchrun tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=torch \
        --device=cuda --comm_backend=nccl --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
    # cuda jit
    echo "  [CUDA][jit backend] Running simulation with dt=${delta_t}"
    # nsys profile  --stats=true  --trace=cuda,osrt,nvtx,openmp,mpi,cublas    --cudabacktrace=all  --force-overwrite=true  --delay 52 
    torchrun tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=cuda \
        --device=cuda --comm_backend=nccl --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
done
