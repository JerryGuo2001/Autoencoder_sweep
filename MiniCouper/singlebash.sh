#!/bin/bash
b=1
for((callback=0; callback<1; callback+=1)); do 
    a=$(callback) 
    for((partition=0; partition<70; partition+=1)); do 
        sbatch file_startup.sub "$partition" "$b" "$a"
        b=$((b + 1))
    done; 
done;