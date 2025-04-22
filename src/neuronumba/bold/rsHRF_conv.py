


def sample():
    
    bold_signals = []
    bold_times = []
    # for step in range(0 + 1, 0 + n_steps + 1):   
    for step in range(1, n_steps + 1):    
    # EVERY STEP: captures the neural activity at each integration step, extract out variable of interest
    # find position within interim buffer where this current step should store data
    # % gives remainder when dividing by interim_istep (6) so will cycle 0 1 2 3 4 5 0 1
    # but bc of -1: will cycle -1 0 1 2 3 4 -1 (-1 means last position in the buffer - wrapping around)
        bold_monitor._interim_stock[((step % bold_monitor._interim_istep) - 1), 0, :, 0] = signal_0_data[step-1, :]
    
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
        # bold = np.expand_dims(np.tensordot(bold_monitor._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)
            for i in range(hrf.shape[0]):                                    # convolving for all the nodes separately
                if i == 0: 
                    bold = np.expand_dims(np.tensordot(bold_monitor._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)
                    # (1, 1, 1)
                else:
                    bold = np.vstack((bold, np.expand_dims(np.tensordot(bold_monitor._stock.transpose(1, 2, 0, 3)[:,i,:,:], hrf[i], axes=([1], [0])), axis = 0)))
                    # now bold is (108, 1, 1)
            
            bold = bold.transpose(1, 0, 2) # now bold is (1, 108, 1)
            
            bold_signals.append(bold)
            bold_times.append(time)
            
            #bold = bold.reshape(bold_monitor._stock.shape[1:]) # but bold is already this?
            result = [time, bold]
