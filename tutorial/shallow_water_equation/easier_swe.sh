#! /bin/bash
export EASIER_DISABLE_JIT_HASH=1
THREADS=20
INTEROP_THREADS=1
export OMP_NUM_THREADS=${THREADS}
export OMP_INTEROP_THREADS=${INTEROP_THREADS}
N_CPU=(500 1000 2000 3000 4000 5000)
N_GPU=(500 750 1000 1250 1500 1750 2000 2250 2500)

# data genereation
for n in ${N_CPU[@]}
do
    if [ ! -f ~/.easier/triangular_${n}.hdf5 ]; then
        python tutorial/create_triangular_mesh.py ${n} ~/.easier/
        torchrun tutorial/shallow_water_equation/assemble_shallow_water.py \
            --device=cuda --backend torch --comm_backend nccl \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5
    fi
done
for n in ${N_GPU[@]}
do
    if [ ! -f ~/.easier/triangular_${n}.hdf5 ]; then
        python tutorial/create_triangular_mesh.py ${n} ~/.easier/
        torchrun tutorial/shallow_water_equation/assemble_shallow_water.py \
            --device=cuda --backend torch --comm_backend nccl \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5
    fi
done

for n in ${N_GPU[@]}
do
    if [ ! -f ~/.easier/SW_${n}.hdf5 ]; then
        python tutorial/create_triangular_mesh.py ${n} ~/.easier/
        torchrun tutorial/shallow_water_equation/assemble_shallow_water.py \
            --device=cuda --backend torch --comm_backend nccl \
            ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5
    fi
done

for n in ${N_GPU[@]}
do
    delta_t=$(echo "scale=8; 0.5/$n" | bc)
    # cuda torch
    torchrun tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=torch \
        --device=cuda --comm_backend=nccl --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
    # cuda jit
    torchrun tutorial/shallow_water_equation/swe_main.py \
        --profile=True --dt=${delta_t} --backend=cuda \
        --device=cuda --comm_backend=nccl --output=res/ \
        ~/.easier/triangular_${n}.hdf5 ~/.easier/SW_${n}.hdf5 
done