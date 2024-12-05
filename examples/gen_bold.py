import argparse
import os
import time

import numpy as np

from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.models import Hopf, Deco2014
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.tools import filterps, hdf
from neuronumba.tools.filters import BandPassFilter
from neuronumba.tools.loader import load_2d_matrix

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Model to use (Hopf, Deco2014)", type=str, default='Deco2014')
    parser.add_argument("--g", help="Global coupling", type=float, default=1.0)
    parser.add_argument("--tmax", help="Simulation time (ms)", type=float, default=10000.0)
    parser.add_argument("--out-path", help="Path to folder for output results", type=str, required=True)
    parser.add_argument("--sc-matrix", help="Path to SC matrix (Matlab, csv, npy, npz)", type=str, required=True)
    parser.add_argument("--sc-index", help="Index inside SC matrix file to use (Matlab and npz only)", type=str, required=False)

    args = parser.parse_args()

    # -------------------------- Load SC matrix
    mat0 = load_2d_matrix(args.sc_matrix, index=args.sc_index)
    sc_norm = 0.2 * mat0 / mat0.max()

    dt = 0.1
    if args.model == 'Deco2014':
        model = Deco2014(g=args.g)
        integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2])
        obs_var = 're'
    else:
        raise RuntimeError(f"Model <{args.model}> not supported!")

    n_rois = sc_norm.shape[0]
    sampling_period = 1.0
    lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    speed = 1.0
    con = Connectivity(weights=sc_norm, lengths=lengths, speed=speed)

    # coupling = CouplingLinearDense(weights=weights, delays=con.delays, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)
    history = HistoryNoDelays()
    # mnt = TemporalAverage(period=1.0, dt=dt)
    monitor = RawSubSample(period=sampling_period, monitor_vars=model.get_var_info([obs_var]))
    s = Simulator(connectivity=con, model=model, history=history, integrator=integ, monitors=[monitor])
    start_time = time.perf_counter()
    s.run(0, args.tmax)
    t_sim = time.perf_counter() - start_time
    data = monitor.data(obs_var)
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(data.shape[0]), data)
    # plt.show()
    b = BoldStephan2008()
    signal = b.compute_bold(data, monitor.period)
    np.save(os.path.join(args.out_path, "bold.npy"), signal)


