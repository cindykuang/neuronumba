#!/bin/bash

# Activate the conda environment
source /home/cindy/miniconda3/etc/profile.d/conda.sh  
conda activate tvb_run

# Change to the base directory
cd /mnt/storage/neuronumba/demos/2_MIST_parcellations

# parcellations to loop through
parcellations=("031-MIST" "056-MIST" "103-MIST" "167-MIST") 

BASE_DIR="/mnt/storage/neuronumba/demos/2_MIST_parcellations"

# Loop through each parcellation
for parc in "${parcellations[@]}"; do
    # Create unique directories for each parcellation
    DATA_RAW_DIR="${BASE_DIR}/${parc}/Data_Raw_${parc}"
    DATA_PRODUCED_DIR="${BASE_DIR}/${parc}/Data_Produced_${parc}"
    
    # Create output directories if they don't exist
    mkdir -p "${DATA_RAW_DIR}"
    mkdir -p "${DATA_PRODUCED_DIR}"
    
    # Determine number of threads based on parcellation
    if [[ "${parc}" == *"031-"* ]]; then
        threads=3  
    elif [[ "${parc}" == *"103-"* ]]; then
        threads=2
    elif [[ "${parc}" == *"167-"* ]]; then
        threads=1
    else
        threads=2  # Use 2 threads for all other parcellations
    fi
        
    # Run the Python script with parcellation-specific paths
    echo "Running trial for parcellation: ${parc}"
    PYTHONPATH="${BASE_DIR}" python3 global_coupling_fitting.py \
        --nproc ${threads} \
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
