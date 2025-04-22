import numpy as np
from numba import njit
import scipy.signal as signal
import os

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.bold.base_bold import Bold


class BoldRegionSpecificHRF(Bold):
    """
    BOLD model using region-specific hemodynamic response functions.
    
    """
    
    hrf_filename = Attr(default=None, required=False)  # Path to file containing pre-computed HRFs
    hrf_data = Attr(default=None, required=False)      # Pre-loaded HRF data
    t_min = Attr(default=20.0, required=False)         # Discard first t_min seconds from signal
    
    def configure(self):
        """Configure the BOLD model by loading HRF data if necessary."""
        super().configure()
        
        # If HRF data is not provided directly but a filename is, load from file
        if self.hrf_data is None and self.hrf_filename is not None:
            if not os.path.exists(self.hrf_filename):
                raise FileNotFoundError(f"HRF file {self.hrf_filename} not found!")
            
            # Load HRF data from file
            # Assuming the file contains a numpy array with shape [n_regions, n_timepoints]
            self.hrf_data = np.loadtxt(self.hrf_filename)
            
        if self.hrf_data is None:
            raise ValueError("Either hrf_data or hrf_filename must be provided!")
            
        # Ensure HRF data is properly shaped
        if len(self.hrf_data.shape) != 2:
            raise ValueError(f"HRF data must be 2D [n_regions, n_timepoints], but got shape {self.hrf_data.shape}")
            
    def compute_bold(self, signal, dt):
        """
        Compute BOLD signal by convolving neural signal with region-specific HRFs.
        
        Parameters:
        -----------
        signal : numpy.ndarray
            Neural activity signal with shape [timepoints, regions]
        dt : float
            Time step in milliseconds
            
        Returns:
        --------
        numpy.ndarray
            BOLD signal with shape [timepoints, regions]
        """
        # Convert dt from ms to seconds 
        dt_sec = dt / 1000.0
        
        # Calculate points to discard from beginning (warmup period)
        n_min = int(np.round(self.t_min / dt_sec))
        
        # Prepare efficient convolution function using numba
        convolve_with_hrf = self._prepare_convolution_function()
        
        # Apply convolution to each region
        bold_signal = convolve_with_hrf(signal, self.hrf_data)
        
        # Discard warmup period
        bold_signal = bold_signal[n_min:, :]
        
        # Resample to TR
        step = int(np.round(self.tr / dt))
        bold_signal = bold_signal[step-1::step, :]
        
        return bold_signal
    
    def _prepare_convolution_function(self):
        """Prepare a numba-optimized function for convolution."""
        
        @njit
        def convolve_with_hrf(neural_signal, hrf_data):
            """
            Convolve neural signal with HRF for each region.
            
            Parameters:
            -----------
            neural_signal : numpy.ndarray
                Neural signal with shape [timepoints, regions]
            hrf_data : numpy.ndarray
                HRF data with shape [regions, hrf_length]
                
            Returns:
            --------
            numpy.ndarray
                BOLD signal with shape [timepoints, regions]
            """
            n_timepoints, n_regions = neural_signal.shape
            
            # Initialize output array
            bold_signal = np.zeros((n_timepoints, n_regions))
            
            # For each region, convolve neural signal with region-specific HRF
            for region in range(n_regions):
                # Get region-specific HRF
                hrf = hrf_data[region, :]
                
                # Reverse HRF for convolution
                hrf = hrf[::-1]
                
                # Convolve neural signal with HRF for this region
                # Using 'same' mode to maintain the original signal length
                bold_signal[:, region] = np.convolve(neural_signal[:, region], hrf, mode='full')[:n_timepoints]
                
            return bold_signal
            
        return convolve_with_hrf
