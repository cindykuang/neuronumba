import numpy as np
from numba import njit

from neuronumba.basic.attr import Attr
from neuronumba.bold.base_bold import Bold
from scipy.special import gamma as sp_gamma

class BoldGlover1999(Bold):

    period = Attr(default=720.0, required=True) #timestep to be convolved to (actual TR)

    hrf_length = Attr(default=24000, required=True)
    
    number_of_nodes = Attr(default=108, required=True)
    
    dt = Attr(default=0.72, required=True) #actual sampling rate of input signal here
    	
    voi = Attr(default=[0], required=True)

    _interim_period = None
    _interim_istep = None
    _interim_stock = None
    _stock_steps = None
    _stock_time = None
    _stock_sample_rate = 2 ** -2 #ms
    hemodynamic_response_function = None
    
    #a1 = Attr(default=6.0, required=False)
    #a2 = Attr(default=13.0, required=False)
    #lam = Attr(default=1.0, required=False)
    #c = Attr(default=0.4, required=False)
    
    
    def config_for_sim(self):
           
        self._stock_sample_rate = 2.0**-2 #/ms   # this is 0.25 , means ONE sample every 4 ms
       	magic_number = self.hrf_length #24000.0
       	#Length of history needed for convolution in steps @ _stock_sample_rate
       	required_history_length = self._stock_sample_rate * magic_number # 6000.0
       	self._stock_steps = np.ceil(required_history_length).astype(int) # 6000
       	stock_time_max    = magic_number/1000.0   # 24
       	stock_time_step   = stock_time_max / self._stock_steps      # 0.004   
       	self._stock_time  = np.arange(0.0, stock_time_max, stock_time_step) # this is an array of (6000,) of timesteps that are 0.004 each, 24s true time total
       	
       	self._interim_period = 1.0 / self._stock_sample_rate #4 # so every 4 steps of the stock sample rate 
       	self._interim_istep = int(round(self._interim_period / self.dt)) # interim period in integration time 		steps (my actual simulation) #every 6 steps of my actual sim at that res
       	
       	self.istep = self.period / self.dt # 1000 # it takes every 1000th step
       	
       	sample_shape = self.voi.shape[0], self.number_of_nodes, 1
       	self._interim_stock = np.zeros((self._interim_istep,) + sample_shape) # (6, 1, 108, 1)
       	self._stock = np.zeros((self._stock_steps,) + sample_shape) # (6000, 1, 108, 1)


    def compute_hrf(self, a1=6.0, a2=13.0, l=1.0, c=0.4):
        """
        Glover (1999) double-gamma HRF approximation.
        t   : array of time points 
        a1  : shape (peak) = 6
        a2  : shape (undershoot) = 13
        l : rate parameter (1/scale) = 1
        c   : mixture coefficient = 0.4
        """
        # compute gamma function values
        gamma_a1 = sp_gamma(a1) #120.0
        gamma_a2 = sp_gamma(a2) #479001600.0
        
        # already have computed array of time points
        t = self._stock_time #(6000,)
        
        # compute the MixtureOfGammas HRF - #(6000,)
        hrf = ((l * t)**(a1-1) * np.exp(-l*t) / gamma_a1) - \
        (c * (l*t)**(a2-1) * np.exp(-l*t) / gamma_a2)
      
        # reverse the HRF to prepare for convolution
        hrf = hrf[::-1]
        
        # adds extra dimension so hrf is (1, 6000) then np.tile() replicates it across the # of nodes
        # so each row is the hrf for 1 node, but they are all identical
        hrfa = np.tile(hrf[np.newaxis, :], (self.number_of_nodes, 1))
        
        self.hemodynamic_response_function = hrfa # gives (108, 6000) just like rsHRF!

    def compute_bold(self, signal, dt):
    
        def Bold_Glover_compute_bold(signal, dt):
            n_steps = signal.shape[0]
            bold_signals = []
            bold_times = []
            
            # for step in range(0 + 1, 0 + n_steps + 1):   
            for step in range(1, n_steps + 1):    
            # EVERY STEP: captures the neural activity at each integration step, extract out variable of interest
            # find position within interim buffer where this current step should store data
            # % gives remainder when dividing by interim_istep (6) so will cycle 0 1 2 3 4 5 0 1
            # but bc of -1: will cycle -1 0 1 2 3 4 -1 (-1 means last position in the buffer - wrapping around)
                self._interim_stock[((step % self._interim_istep) - 1), 0, :, 0] = signal[step-1, :]
            
            # EVERY 6 (interim_istep) STEPS: compute an average of the neural activity in this interim window and update the MAIN STOCK
                if step % self._interim_istep == 0:
                    avg_interim_stock = np.mean(self._interim_stock, axis=0)
                    self._stock[((step//self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock
                    # Stores this downsampled activity in the stock buffer used for convolution
                    
            # EVERY 1000 (istep) STEPS:    
                if step % self.istep == 0:
                    time = step * self.dt # true time, given in ms
                    hrf = np.roll(self.hemodynamic_response_function,
                                     ((step//self._interim_istep % self._stock_steps) - 1), # this is just stock buffer position
                                     axis=1)
                # hrf is shape (108, 6000)
                # transposes stock array: orig dimensions were (timesteps, state_vars, brain regions, modes) 
                # or (6000, 1, 108, 1)
                # but now reordered to: (state_vars, brain regions, timesteps, modes)
                
                    # convolve each ROI separately
                    for i in range(hrf.shape[0]):                                    
                        if i == 0: 
                            bold = np.expand_dims(np.tensordot(self._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)
                            # (1, 1, 1)
                        else:
                            bold = np.vstack((bold, np.expand_dims(np.tensordot(self._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)))
                            # at the end bold is (108, 1, 1)
                    
                    bold = bold.transpose(1, 0, 2) # now bold is (1, 108, 1)
                    
                    bold_signals.append(bold) # save this for all timepoints
                    bold_times.append(time)
                    
            bold_signals = np.array(bold_signals)  # Should be shape (num_timepoints, 1, 108, 1) # (1227, 1, 108, 1)
            bold_times = np.array(bold_times)    # (1227,)
            
            bold_signals_truncated = bold_signals[35:, :, :, :] # now ti is (1200, 1, 108, 1) # TODO - hard coded
            bold_signals_2d = bold_signals_truncated.reshape(1200, 108) # TODO
                    
            bold = bold.reshape(self._stock.shape[1:]) # but bold is already this?
            return bold_signals_2d
                
                
        bds = Bold_Glover_compute_bold(signal, dt=dt)
        #step = int(np.round(self.tr / dt))  # each step is the length of the TR, in milliseconds
        #bds = b[step - 1::step, :] # my bold is already downsampled though right? maybe trim here # TODO
        return bds
