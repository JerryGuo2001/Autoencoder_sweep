#!/bin/bash
for((callback=0; callback<1; callback+=1)); do 
    a=$(callback) 
    for((partition=0; partition<70; partition+=1)); do 
        b=$(a*partition)
        sbatch file_startup.sub "$partition" "$b" "$a"; 
    done; 
done;