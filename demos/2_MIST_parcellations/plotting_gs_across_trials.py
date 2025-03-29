#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 11:53:43 2025

@author: cindy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from neuronumba.observables.measures import PearsonSimilarity


base_path = "/mnt/storage/neuronumba/demos/2_MIST_parcellations/"
resolutions = ['031', '056', '103', '167']
observable_name = "FC" 
g_values = {res: [] for res in resolutions}
correlations = {res: [] for res in resolutions}

fig, ax = plt.subplots(figsize=(10, 6))

ax.set(xlabel='G (global coupling)', 
       ylabel=f'Pearson Similarity ({observable_name})',
       title='Global Coupling Parameter Fitting')

ax.grid(True)

colors = ['blue', 'green', 'orange', 'red']
for i, res in enumerate(resolutions):
    out_file_path = os.path.join(base_path, f"{res}-MIST/Data_Produced_{res}-MIST/Deco2014_FC")
    
    # Get all fitting_g*.mat files
    
    # List all files in the directory
    files = os.listdir(out_file_path)
    fitting_files = [f for f in files if f.startswith("fitting_g") and f.endswith(".mat")]
    
    for file in fitting_files:
        # Extract g value from filename
        g_str = file.replace("fitting_g", "").replace(".mat", "")
        g = float(g_str)
        
        # Load data
        data_path = os.path.join(out_file_path, file)
        with h5py.File(data_path, 'r') as f:        
            sim_fc = f[observable_name][:]
            
        # Load empirical data
        emp_path = os.path.join(out_file_path, "fNeuro_emp.mat")
        with h5py.File(emp_path, 'r') as f:        
            emp_fc = f[observable_name][:]
        
        dist = PearsonSimilarity().distance(sim_fc, emp_fc)
        
        g_values[res].append(g)
        correlations[res].append(dist)

    # Sort by g value
    sorted_indices = np.argsort(g_values[res]) 
    sorted_g = [g_values[res][i] for i in sorted_indices]
    sorted_corr = [correlations[res][i] for i in sorted_indices]
    
    ax.plot(sorted_g, sorted_corr, 'o-', label=f'{res}-MIST', color=colors[i])
    
    max_corr = max(sorted_corr)
    max_idx = sorted_corr.index(max_corr)
    ax.annotate(f'Max: {max_corr:.3f}',
                (sorted_g[max_idx], max_corr),
                textcoords="offset points",
                xytext=(5,-36),
                ha='center',
                arrowprops=dict(arrowstyle='->'))

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(base_path, f'all_parcellations_fitting_{observable_name}.png'), dpi=300, bbox_inches="tight")
plt.show()