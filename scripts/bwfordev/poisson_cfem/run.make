#!/bin/bash

target=$1

for nodes in "4" "8" "16" "32" "64";
do
    # maximum of 16 procs per node
    for ppn in "2" "4" "8" "16"
    do
    processes=$(($nodes * $ppn))
    echo "nodes=$nodes ppn=$ppn processes=$processes"
    sed "s/&NODES&/$nodes/g" run.base | sed "s/&PPN&/$ppn/g" | sed "s/&MYPROG&/$target/g" > run.${target}.${nodes}.${ppn}
    done
done
