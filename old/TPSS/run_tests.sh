#!/bin/bash

## run this script in the build directory

declare -a arrTests
for file in $(pwd)/TPSS/T*
do
    # echo $file
    arrTests=("${arrTests[@]}" "$file")
done

echo “number of tests ${#arrTests[@]}”
for file in "${arrTests[@]}"
do
    echo -e "\e[32mexecute *** ${file} ***\e[0m" # set font color \e[32m and reset format by \e[0m
    str_file=${file}
    eval "$file" &> "$(basename "${str_file}").log"
done
