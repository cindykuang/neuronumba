import numpy as np
import numba as nb
import math

from neuronumba.basic.attr import HasAttr, Attr
from neuronumba.numba_tools import addr
from neuronumba.numba_tools.addr import address_as_void_pointer
from neuronumba.numba_tools.types import NDA_f8_2d

class Monitor(HasAttr):
    dt = Attr(required=True)
    n_rois = Attr(required=True)
    
    monitor_vars = Attr(required=True)
    state_vars_indices = Attr(dependant=True)
    obs_vars_indices = Attr(dependant=True)

    n_state_vars = Attr(dependant=True)
    n_obs_vars = Attr(dependant=True)

    def _init_dependant(self):
        # Extract index list from input dictionary
        s_vars_i = []
        o_var_i = []

        for v in self.monitor_vars:
            if self.monitor_vars[v][0]:
                s_vars_i.append(self.monitor_vars[v][2])
            else:
                o_var_i.append(self.monitor_vars[v][2])

        self.n_state_vars = len(s_vars_i)
        self.n_obs_vars = len(o_var_i)
        self.state_vars_indices = np.array(s_vars_i, dtype=np.int32)
        self.obs_vars_indices = np.array(o_var_i, dtype=np.int32)

    def data(self, var: str):
        if var not in self.monitor_vars:
            raise Exception(f"Variable <{var}> not defined in monitor!")
        value = self.monitor_vars[var]
        if value[0]:
            return self._get_data_state(value[1])
        else:
            return self._get_data_obs(value[1])

    # Methods to be implemented in subclasses

    def sample(self, step, state, observed):
        raise NotImplementedError

    def _get_data_state(self, index: int):
        raise NotImplementedError

    def _get_data_obs(self, index: int):
        raise NotImplementedError


class RawMonitor(Monitor):

    buffer = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.buffer = []

    def sample(self, step, state, observed):
        self.buffer.append(state)

    def data(self):
        return np.array(self.buffer)


class RawSubSample(Monitor):
    period = Attr(default=None, required=True)
    t_max = Attr(default=None, required=True)

    n_interim_samples = Attr(dependant=True)
    buffer_state = Attr(dependant=True)
    buffer_observed = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_interim_samples = int(self.period / self.dt)
        n_steps = int(self.t_max / self.dt)
        time_samples = 1 + int(n_steps / self.n_interim_samples)
        if self.n_state_vars:
            self.buffer_state = np.zeros((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.buffer_observed = np.zeros((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.buffer_observed = np.empty(1, )

    def _get_data_state(self, index: int):
        return self.buffer_state[:, index, :]

    def _get_data_obs(self, index: int):
        return self.buffer_observed[:, index, :]

    def get_numba_sample(self):
        bs = self.buffer_state
        bs_addr, bs_shape, bs_dtype = addr.get_addr(bs)
        state_vars = self.state_vars_indices
        n_state = nb.intc(self.n_state_vars)
        bo = self.buffer_observed
        bo_addr, bo_shape, bo_dtype = addr.get_addr(bo)
        obs_vars = self.obs_vars_indices
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step: nb.intc, state: NDA_f8_2d, observed: NDA_f8_2d):
            if step % n_interim_samples == 0:
                if n_state > 0:
                    bs = nb.carray(address_as_void_pointer(bs_addr), bs_shape, dtype=bs_dtype)
                    i = int(step / n_interim_samples)
                    for v in range(n_state):
                        bs[i, v, :] = state[state_vars[v], :]
                if n_obs > 0:
                    bo = nb.carray(address_as_void_pointer(bo_addr), bo_shape, dtype=bo_dtype)
                    i = int(step / n_interim_samples)
                    for v in range(n_obs):
                        bo[i, v, :] = observed[obs_vars[v], :]

        return m_sample


class TemporalAverage(Monitor):
    period = Attr(default=None, required=True)
    t_max = Attr(default=None, required=True)

    n_interim_samples = Attr(dependant=True)
    buffer_state = Attr(dependant=True)
    buffer_observed = Attr(dependant=True)

    i_buffer_state = Attr(dependant=True)
    i_buffer_observed = Attr(dependant=True)

    def _init_dependant(self):
        super()._init_dependant()
        self.n_interim_samples = int(self.period / self.dt)
        # Okey this give problems, I think the +1 is not ok when resto modulo is zero.
        # Because I'm not sure I can remove it, I will put the math.ceil. If this cause problems
        # in divisions (ex. t_max / perios) that are not exact, remove the ceil and leave just the
        # int(self.t_max / self.perios). 
        # The following was the line causing problems:
        # time_samples = 1 + int(self.t_max / self.period)
        # And this the proposal:
        time_samples = math.ceil(self.t_max / self.period)
        if self.n_state_vars:
            self.i_buffer_state = np.zeros((self.n_interim_samples, self.n_state_vars, self.n_rois))
            self.buffer_state = np.empty((time_samples, self.n_state_vars, self.n_rois))
        else:
            self.i_buffer_state = np.empty(1, )
            self.buffer_state = np.empty(1, )
        if self.n_obs_vars:
            self.i_buffer_observed = np.zeros((self.n_interim_samples, self.n_obs_vars, self.n_rois))
            self.buffer_observed = np.empty((time_samples, self.n_obs_vars, self.n_rois))
        else:
            self.i_buffer_observed = np.empty(1, )
            self.buffer_observed = np.empty(1, )

    def data_state(self):
        return self.buffer_state

    def data_observed(self):
        return self.buffer_observed

    def _get_data_state(self, index: int):
        return self.buffer_state[:, index, :]

    def _get_data_obs(self, index: int):
        return self.buffer_observed[:, index, :]

    def get_numba_sample(self):
        buffer_state = self.buffer_state
        bs_addr, bs_shape, bs_dtype = addr.get_addr(buffer_state)
        i_buffer_state = self.i_buffer_state
        ibs_addr, ibs_shape, ibs_dtype = addr.get_addr(i_buffer_state)
        state_vars = self.state_vars_indices
        n_state = nb.intc(self.n_state_vars)
        buffer_observed = self.buffer_observed
        bo_addr, bo_shape, bo_dtype = addr.get_addr(buffer_observed)
        i_buffer_observed = self.i_buffer_observed
        ibo_addr, ibo_shape, ibo_dtype = addr.get_addr(i_buffer_observed)
        obs_vars = self.obs_vars_indices
        n_obs = nb.intc(self.n_obs_vars)
        n_interim_samples = nb.intc(self.n_interim_samples)

        @nb.njit(nb.void(nb.intc, nb.f8[:, :], nb.f8[:, :]))
        def m_sample(step, state, observed):
            # Update interim buffer
            if n_state > 0:
                # i_bnb_state = nb.carray(address_as_void_pointer(ibs_addr), ibs_shape, dtype=ibs_dtype)
                i_bnb_state = addr.create_carray(ibs_addr, ibs_shape, ibs_dtype)
                # i_bnb_state = self.i_buffer_state
                i_bnb_state[(step - 1) % n_interim_samples] = state[state_vars, :]
                if step % n_interim_samples == 0:
                    # bnb_state = nb.carray(address_as_void_pointer(bs_addr), bs_shape, dtype=bs_dtype)
                    bnb_state = addr.create_carray(bs_addr, bs_shape, bs_dtype)
                    # bnb_state = self.buffer_state
                    i = nb.intc(step / n_interim_samples)
                    bnb_state[i, :, :] = i_bnb_state.sum(axis=0) / ibs_shape[0]

            if n_obs > 0:
                # i_bnb_observed = nb.carray(address_as_void_pointer(ibo_addr), ibo_shape, dtype=ibo_dtype)
                i_bnb_observed = addr.create_carray(ibo_addr, ibo_shape, ibo_dtype)
                # i_bnb_observed = self.i_buffer_observed
                i_bnb_observed[(step - 1) % n_interim_samples] = observed[obs_vars, :]
                if step % n_interim_samples == 0:
                    # bnb_observed = nb.carray(address_as_void_pointer(bo_addr), bo_shape, dtype=bo_dtype)
                    bnb_observed = addr.create_carray(bo_addr, bo_shape, bo_dtype)
                    # bnb_observed = self.buffer_observed
                    i = nb.intc(step / n_interim_samples)
                    bnb_observed[i, :, :] = i_bnb_observed.sum(axis=0) / ibo_shape[0]

        return m_sample
        
        
class Bold(Monitor):
    """

    Base class for the Bold monitor.

    **Attributes**

        hrf_kernel: the haemodynamic response function (HRF) used to compute
                    the BOLD (Blood Oxygenation Level Dependent) signal.

        length    : duration of the hrf in seconds.

        period    : the monitor's period

    **References**:

    .. [B_1997] Buxton, R. and Frank, L., *A Model for the Coupling between
        Cerebral Blood Flow and Oxygen Metabolism During Neural Stimulation*,
        17:64-72, 1997.

    .. [Fr_2000] Friston, K., Mechelli, A., Turner, R., and Price, C., *Nonlinear
        Responses in fMRI: The Balloon Model, Volterra Kernels, and Other
        Hemodynamics*, NeuroImage, 12, 466 - 477, 2000.

    .. [Bo_1996] Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David
        J. Heeger (1996). Linear Systems Analysis of Functional Magnetic Resonance
        Imaging in Human V1. J Neurosci 16: 4207-4221

    .. [Po_2000] Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
        (2000). Neuronal activity in human primary visual cortex correlates with
        perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    .. [Gl_1999] Glover, G. *Deconvolution of Impulse Response in Event-Related BOLD fMRI*.
        NeuroImage 9, 416-429, 1999.

    .. note:: gamma and polonsky are based on the nitime implementation
              http://nipy.org/nitime/api/generated/nitime.fmri.hrf.html

    .. note:: see Tutorial_Exploring_The_Bold_Monitor

    """
    _ui_name = "BOLD"

    period = Float(
        label="Sampling period (ms)",
        default=2000.0,
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
        default=20000.,
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
        else :
            #Reverse it, need it into the past for matrix-multiply of stock
            G = G[::-1]
            self.hemodynamic_response_function = G[numpy.newaxis, :]
        #Interim stock configuration
        self._interim_period = 1.0 / self._stock_sample_rate #period in ms
        self._interim_istep = int(round(self._interim_period / self.dt)) # interim period in integration time steps
        self.log.debug('Bold HRF shape %s, interim period & istep %d & %d',
                  self.hemodynamic_response_function.shape, self._interim_period, self._interim_istep)                                                       

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

    def sample(self, step, state):
        # Update the interim-stock at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi, :]
        # At stock's period update it with the temporal average of interim-stock
        if step % self._interim_istep == 0:
            avg_interim_stock = numpy.mean(self._interim_stock, axis=0)
            self._stock[((step//self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock
        # At the monitor's period, apply the heamodynamic response function to
        # the stock and return the resulting BOLD signal.
        if step % self.istep == 0:
            time = step * self.dt
            hrf = numpy.roll(self.hemodynamic_response_function,
                             ((step//self._interim_istep % self._stock_steps) - 1),
                             axis=1)
            if isinstance(self.hrf_kernel, equations.RestingStateHRF):           # rsHRF has been obtained from a file
                for i in range(hrf.shape[0]):                                    # convolving for all the nodes separately
                    if i == 0:
                        bold = numpy.expand_dims(numpy.tensordot(self._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)
                    else:
                        bold = numpy.vstack((bold, numpy.expand_dims(numpy.tensordot(self._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)))
                bold = bold.transpose(1, 0, 2)
            elif isinstance(self.hrf_kernel, equations.FirstOrderVolterra):
                k1_V0 = self.hrf_kernel.parameters["k_1"] * self.hrf_kernel.parameters["V_0"]
                bold = (numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3))) - 1.0) * k1_V0
            else:
                bold = numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3)))
            bold = bold.reshape(self._stock.shape[1:])
            return [time, bold]
