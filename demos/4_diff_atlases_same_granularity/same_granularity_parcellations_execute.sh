#!/bin/bash

# Activate the conda environment
source /home/cindy/miniconda3/etc/profile.d/conda.sh  
conda activate tvb_run

# Change to the base directory
cd /mnt/storage/neuronumba/demos/4_diff_atlases_same_granularity

# parcellations to loop through
parcellations=("096-HarvardOxfordMaxProbThr0" "100-Schaefer17Networks" "108-CraddockSCorr2Level") #"103-MIST" 

BASE_DIR="/mnt/storage/neuronumba/demos/4_diff_atlases_same_granularity"

# Loop through each parcellation
for parc in "${parcellations[@]}"; do
    # Create unique directories for each parcellation
    DATA_RAW_DIR="${BASE_DIR}/${parc}/Data_Raw_${parc}"
    DATA_PRODUCED_DIR="${BASE_DIR}/${parc}/Data_Produced_${parc}"
    
    # Create output directories if they don't exist
    mkdir -p "${DATA_RAW_DIR}"
    mkdir -p "${DATA_PRODUCED_DIR}"
        
    # Run the Python script with parcellation-specific paths
    echo "Running trial for parcellation: ${parc}"
    PYTHONPATH="${BASE_DIR}" python3 global_coupling_fitting.py \
        --nproc 1 \
        --model Deco2014 \
        --observable FC \
        --g-range 1.0 2.4 0.2 \
        --tr 720 \
        --fmri-path "${DATA_RAW_DIR}" \
        --out-path "${DATA_PRODUCED_DIR}" \
        --fic Herzog
    
    echo "Completed trial for ${parc}"
    echo "-------------------"
done

echo "All parcellation trials completed."

conda deactivate
