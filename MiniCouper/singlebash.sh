#!/bin/bash
for((callback=0; callback<1; callback+=1)); do
    for((partition=0; partition<9; partition+=1)); do 
        sbatch file_startup.sub "$partition" "$callback"; 
    done; 
done;