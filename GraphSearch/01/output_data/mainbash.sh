#!/bin/bash
for((callback=1; callback<1; callback+=1)); do  
    for((partition=0; partition<3; partition+=1)); do 
        a=$((callback + partition))
        sbatch file_startup.sub "$partition" "$callback" "$a"; 
    done; 
done;