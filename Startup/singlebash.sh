#!/bin/bash
for((partition=0; partition<3; partition+=1)); do 
    a=$((1 + partition))
    sbatch file_startup.sub "$partition" "$a"; 
done; 