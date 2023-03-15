#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 # number of MPI ranks per node
#SBATCH --gres=gpu:v100:1    # GPUs per node; should equal tasks-per-node
#SBATCH --time=48:00:00
#SBATCH --mem=65536MB
set -euo pipefail # kill the job if anything fails
set -x # echo script
module purge
module load julia/1.8.0 hdf5/1.12.1 cuda/11.2 openmpi/4.1.1_cuda-11.2 # CUDA-aware MPI
export JULIA_NUM_THREADS=${SLURM_GPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'
julia --project -e 'using CUDA; @info CUDA.has_cuda()'
mpiexec julia --project training.jl $1
mpiexec julia --project analysis.jl $1
#mpiexec julia --project analyze_train_by_batch.jl 1 $1
#mpiexec julia --project analyze_train_by_batch.jl 2 $1
#mpiexec julia --project analyze_train_by_batch.jl 4 $1
#mpiexec julia --project analyze_train_by_batch.jl 8 $1
#mpiexec julia --project analyze_train_by_batch.jl 16 $1
# Fields are nbatches, wavenumber, experiment toml
#mpiexec julia --project analyze_by_batch.jl 20 1 $1
#mpiexec julia --project analyze_by_batch.jl 20 2 $1
#mpiexec julia --project analyze_by_batch.jl 20 4 $1
#mpiexec julia --project analyze_by_batch.jl 20 8 $1
#mpiexec julia --project analyze_by_batch.jl 20 16 $1
echo done
