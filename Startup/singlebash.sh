#!/bin/bash
for((partition=0; partition<1; partition+=1)); do 
    sbatch file_startup.sub "$partition"; 
done; 