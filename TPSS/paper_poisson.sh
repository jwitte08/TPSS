#!/bin/bash                                                                                                                           
pwd=$(pwd)
echo $pwd

bash $HOME/.dealii/config_release.sh

declare -a arrSmoother
for variant in "ACP" "MCP" "MVP"
do
    arrSmoother=("${arrSmoother[@]}" "$variant")
done

smoother="${arrSmoother[$1]}"
steps=$2

#for ((degree=start; degree<=end; ++degree))
for degree in 3 7 15
do
    for dim in 2 3
    do
    echo "deg: $degree"
    echo "dim: $dim"
    echo "smo: $smoother"
    python3 ct_parameter.py -DIM $dim -DEG $degree -SMO $smoother
    make -j
    ./TPSS/paper_poisson $steps
    done
done
