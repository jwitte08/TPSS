#!/bin/bash

declare -a refinements=('4' '5' '6' '7' '8')
declare -a myprogs=()
for prog in $@;
do
    myprogs+=("${prog}")
done

nodes=2
ppn=28
args=${refinements[*]}
proglist=${myprogs[*]}

for nodes in "1" "2" "4" "8" "16" "32" "64";
do
processes=$(($nodes * $ppn))
echo "nodes=$nodes ppn=$ppn processes=$processes"

sed "s/&NODES&/$nodes/g" run.r1.base | sed "s/&PPN&/$ppn/g" | sed "s/&MYPROG&/$proglist/g" | sed "s/&ARGS&/$args/g" > run.r1.${nodes}

done
