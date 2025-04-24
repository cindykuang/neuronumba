import numpy as np
from numba import njit

from neuronumba.basic.attr import Attr
from neuronumba.bold.base_bold import Bold

class Bold_rsHRF(Bold):

    period = Float(
        label="Sampling period (ms)",
        default=720.0,
        doc="""For the BOLD monitor, sampling period in milliseconds must be
        an integral multiple of 500. Typical measurment interval (repetition
        time TR) is between 1-3 s. If TR is 2s, then Bold period is 2000ms.""")

    hrf_kernel = Attr(
        equations.HRFKernelEquation,
        label="Haemodynamic Response Function",
        default=equations.FirstOrderVolterra(),
        required=True,
        doc="""A tvb.datatypes.equation object which describe the haemodynamic
        response function used to compute the BOLD signal.""")

    hrf_length = Float(
        label="Duration (ms)",
        default=24000.,
        doc= """Duration of the hrf kernel""",)
        #order=-1)

    _interim_period = None
    _interim_istep = None
    _interim_stock = None
    _stock_steps = None
    _stock_time = None
    _stock_sample_rate = 2 ** -2
    hemodynamic_response_function = None
    
    def compute_hrf(self):
        """
        Compute the hemodynamic response function.

        """
        self._stock_sample_rate = 2.0**-2 #/ms    # NOTE: An integral multiple of dt
        magic_number = self.hrf_length #* 0.8      # truncates G, volterra kernel, once ~zero
        #Length of history needed for convolution in steps @ _stock_sample_rate
        required_history_length = self._stock_sample_rate * magic_number # 3840 for tau_s=0.8
        self._stock_steps = numpy.ceil(required_history_length).astype(int)
        stock_time_max    = magic_number/1000.0                                # [s]
        stock_time_step   = stock_time_max / self._stock_steps                 # [s]
        self._stock_time  = numpy.arange(0.0, stock_time_max, stock_time_step) # [s]
        self.log.debug("Bold requires %d steps for HRF kernel convolution", self._stock_steps)            # if the input has not been obtained from file
        #Compute the HRF kernel
        G = self.hrf_kernel.evaluate(self._stock_time)
        if isinstance(self.hrf_kernel, equations.RestingStateHRF): 
            #rsHRF for each region, reversed and upsampled to self._stock_steps
            self.hemodynamic_response_function = G 
        #else :
            #Reverse it, need it into the past for matrix-multiply of stock
            #G = G[::-1]
            #self.hemodynamic_response_function = G[numpy.newaxis, :]
        #Interim stock configuration
        self._interim_period = 1.0 / self._stock_sample_rate #period in ms
        self._interim_istep = int(round(self._interim_period / self.dt)) # interim period in integration time steps
        #self.log.debug('Bold HRF shape %s, interim period & istep %d & %d',
                  #self.hemodynamic_response_function.shape, self._interim_period, self._interim_istep)                
                  
    def _config_vois(self, simulator):
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]

    def _config_time(self, simulator):
        self.dt = simulator.integrator.dt
        self.istep = ReferenceBackend.iround(self.period / self.dt)

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

        Grab the Simulator's integration step size. Set the monitor's variables
        of interest based on the Monitor's 'variables_of_interest' attribute, if
        it was specified, otherwise use the 'variables_of_interest' specified 
        for the Model. Calculate the number of integration steps (isteps)
        between returns by the record method. This method is called from within
        the the Simulator's configure() method.

        """
        self._config_vois(simulator)
        self._config_time(simulator)                                                

    def config_for_sim(self, simulator):
        super(Bold, self).config_for_sim(simulator)
        self.compute_hrf()
        if isinstance(self.hrf_kernel, equations.RestingStateHRF):                          # if HRF has been obtained from a file
            if self.hemodynamic_response_function.shape[0] != simulator.number_of_nodes:    # if the number of nodes do not match for the input and simulator
                self.log.error("Unexpect File Input! Expected Input of shape: %d, Obtained Input of Shape: %d", simulator.number_of_nodes, self.hemodynamic_response_function.shape[0])
        sample_shape = self.voi.shape[0], simulator.number_of_nodes, simulator.model.number_of_modes
        self._interim_stock = numpy.zeros((self._interim_istep,) + sample_shape)
        self.log.debug("BOLD inner buffer %s %.2f MB" % (
            self._interim_stock.shape, self._interim_stock.nbytes/2**20))
        self._stock = numpy.zeros((self._stock_steps,) + sample_shape)
        self.log.debug("BOLD outer buffer %s %.2f MB" % (
            self._stock.shape, self._stock.nbytes/2**20))
            
# this is the sample method from amogh's bold monitors.py
# def sample():

def compute_bold(self, signal, dt):

    n_steps = signal.shape[0]
    bold_signals = []
    bold_times = []
    
    # for step in range(0 + 1, 0 + n_steps + 1):   
    for step in range(1, n_steps + 1):    
    # EVERY STEP: captures the neural activity at each integration step, extract out variable of interest
    # find position within interim buffer where this current step should store data
    # % gives remainder when dividing by interim_istep (6) so will cycle 0 1 2 3 4 5 0 1
    # but bc of -1: will cycle -1 0 1 2 3 4 -1 (-1 means last position in the buffer - wrapping around)
        bold_monitor._interim_stock[((step % bold_monitor._interim_istep) - 1), 0, :, 0] = signal[step-1, :]
    
    # EVERY 6 (interim_istep) STEPS: compute an average of the neural activity in this interim window and update the MAIN STOCK
        if step % bold_monitor._interim_istep == 0:
            avg_interim_stock = np.mean(bold_monitor._interim_stock, axis=0)
            bold_monitor._stock[((step//bold_monitor._interim_istep % bold_monitor._stock_steps) - 1), :] = avg_interim_stock
            # Stores this downsampled activity in the stock buffer used for convolution
            
    # EVERY 1000 (istep) STEPS:    
        if step % bold_monitor.istep == 0:
            time = step * bold_monitor.dt # true time, given in ms
            hrf = np.roll(bold_monitor.hemodynamic_response_function,
                             ((step//bold_monitor._interim_istep % bold_monitor._stock_steps) - 1), # this is just stock buffer position
                             axis=1)
        # hrf is shape (108, 6000)
        # transposes stock array: orig dimensions were (timesteps, state_vars, brain regions, modes) 
        # or (6000, 1, 108, 1)
        # but now reordered to: (state_vars, brain regions, timesteps, modes)
        
            # convolve each ROI separately
            for i in range(hrf.shape[0]):                                    
                if i == 0: 
                    bold = np.expand_dims(np.tensordot(bold_monitor._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)
                    # (1, 1, 1)
                else:
                    bold = np.vstack((bold, np.expand_dims(np.tensordot(bold_monitor._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)))
                    # at the end bold is (108, 1, 1)
            
            bold = bold.transpose(1, 0, 2) # now bold is (1, 108, 1)
            
            bold_signals.append(bold) # save this for all timepoints
            bold_times.append(time)
            
    bold_signals = np.array(bold_signals)  # Should be shape (num_timepoints, 1, 108, 1) # (1227, 1, 108, 1)
    bold_times = np.array(bold_times)    # (1227,)
    
    bold_signals_truncated = bold_signals[27:, :, :, :] # now ti is (1200, 1, 108, 1) # TODO
    bold_signals_2d = bold_signals_truncated.reshape(1200, 108) # TODO
            
    bold = bold.reshape(bold_monitor._stock.shape[1:]) # but bold is already this?
    return [bold_signals_2d]
            
            
b = Bold_rsHRF.compute_bold(signal, dt=dt)
#step = int(np.round(self.tr / dt))  # each step is the length of the TR, in milliseconds
#bds = b[step - 1::step, :] # my bold is already downsampled though right? maybe trim here # TODO
return bds
