#!/bin/bash                                                                                                                           
pwd=$(pwd)
echo $pwd

bash $HOME/.dealii/config_release.sh

declare -a arrSmoother
for variant in "ACP" "MCP" "MVP"
do
    arrSmoother=("${arrSmoother[@]}" "$variant")
done

start=$1
end=$2
smoother="${arrSmoother[$3]}"
steps=$4
printf 'test %s with %i steps for degrees %i to %i \n' "$smoother" $steps $start $end

for ((degree=start; degree<=end; ++degree))
do
    echo "d: $degree"
    echo "s: $smoother"
    python3 ct_parameter.py -DIM 2 -DEG $degree -SMO $smoother
    for damping in 1.0  0.9  0.8  0.7  0.6
    do
    for lambda in 1.0  5.0  10.0  25.0  50.0  100.0
    do
	mu=1.0
	echo "damping: $damping"
	echo "lambda: $lambda"
	echo "mu: $mu"
	make -j
	./TPSS/proceeding_elasticity_diagonly $steps $damping $mu $lambda
    done
    done
done
