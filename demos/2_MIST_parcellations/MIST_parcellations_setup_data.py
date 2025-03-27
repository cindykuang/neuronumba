#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:12:29 2025

@author: cindy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:47:43 2025

using their MIST rs-fMRI data - already parcellated

"""

import numpy as np
import nibabel as nib
from nilearn import plotting

import os 
from os.path import join as opj
import numpy as np
import shutil
import pandas as pd
os.chdir('/mnt/storage/hcp_clean/code')
import cifti_processing
import functional_connectivity
import utils

""" MAKE SURE DATA_RAW FOLDER IS CLEAN AND READY """

""" LOADING IN PRE-PARCELLATED FMRI DATA AND SC MATRICES """

base_dir = '/mnt/storage/neuronumba/demos/2_MIST_parcellations'
resolutions = ['031', '056', '103', '167']
EBrains_path = '/mnt/storage/neuronumba/EBrains/'
session = 'REST1'
run = 'LR'


current_subs = np.loadtxt('/home/cindy/src/neuronumba/examples/current_subs_n=29.txt', dtype=int)
subj_IDs = np.loadtxt('/home/cindy/src/neuronumba/examples/Popovych_EBrains_Dataset_200_HCP_SubjectIDList.txt')

for res in resolutions:
    
    # make sure data raw folder exists
    target_path = opj(base_dir, f'{res}-MIST', f'Data_Raw_{res}-MIST')
    utils.check_directory_exists(target_path)
    
    # make sure data produced folder exists
    Data_Produced_path = opj(base_dir, f'{res}-MIST', f'Data_Produced_{res}-MIST')
    utils.check_directory_exists(Data_Produced_path)
    
    for sub in current_subs:
        
        # create specific subject folders
        sub_folder = opj(target_path, f'sub-{sub}')
        utils.check_directory_exists(sub_folder)
            
        index = np.where(subj_IDs==sub)[0][0]
        index = f"{index:03d}" 
        
        # now get the sc matrices from EBrains downloaded 200Schaefer17Networks dataset (unzipped)
        if os.path.isdir(opj(EBrains_path, 'structural', f'{res}-MIST', '1StructuralConnectivity', index)):
            shutil.copyfile(opj(EBrains_path, 'structural', f'{res}-MIST', '1StructuralConnectivity', index, 'Counts.csv'),
                            opj(target_path, f'sub-{sub}', 'Counts.csv'))
            
        # get the pre-parcellated data and convert to .csv (weird spacing issues)?
        if os.path.isdir(opj(EBrains_path, 'parcellated_rsfMRI', f'{res}-MIST', index)):
            # shutil.copyfile(opj(EBrains_path, 'parcellated_rsfMRI', f'{res}-MIST', index, f'rfMRI_{session}_{run}_BOLD.tsv'),
            #                 opj(target_path, f'sub-{sub}', f'rfMRI_{session}_{run}_BOLD.tsv'))
            array = np.loadtxt(opj(EBrains_path, 'parcellated_rsfMRI', f'{res}-MIST', index, f'rfMRI_{session}_{run}_BOLD.tsv'))
            np.savetxt(opj(target_path, f'sub-{sub}', f'rfMRI_{session}_{run}_BOLD.csv'), array)
    
                    



