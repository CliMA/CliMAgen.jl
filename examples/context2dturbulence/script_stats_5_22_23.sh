#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 # number of MPI ranks per node
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
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
# Fields are nbatches, wavenumber, experiment toml
#mpiexec julia --project analyze_by_batch.jl 31 2000 16 experiments/Experiment_all_data_centered_dropout_05.toml output/all_data_centered_dropout_05/epoch_125 stats_5_22_23/baseline false
#mpiexec julia --project analyze_by_batch.jl 31 2000 16 experiments/Experiment_all_data_centered_dropout_05.toml output/all_data_centered_dropout_05/epoch_125 stats_5_22_23/baseline_smooth true
#mpiexec julia --project analyze_by_batch.jl 31 2000 16 experiments/Experiment_single_wavenumber_centered_dropout_05.toml output/single_wavenumber_centered_dropout_05 stats_5_22_23/single_wn_1e-2 false
mpiexec julia --project analyze_by_batch.jl 4 2000 16 experiments/Experiment_single_wavenumber_centered_dropout_05_sigma_min.toml output/single_wavenumber_centered_dropout_05_sigma_min stats_5_22_23/single_wn_1e-3 false


echo done
