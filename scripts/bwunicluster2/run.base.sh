#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=&NODES&
#SBATCH --ntasks-per-node=&PPN&
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julius.witte@iwr.uni-heidelberg.de

export OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1
module load mpi/openmpi/4.0

echo "starting on `date` with ppn=&PPN&, nodes=&NODES& at `pwd`"

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "DEAL_II_NUM_THREADS=$DEAL_II_NUM_THREADS"

for prog in &MYPROG&;
do
for args in &ARGS&;
do
    cmd="./${prog} ${args}"
    mpirun --bind-to core --map-by core -display-devel-map -report-bindings ${cmd}
done
done

echo "exiting on `date` with ppn=&PPN&, nodes=&NODES& at `pwd`"
