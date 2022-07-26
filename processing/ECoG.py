import numpy as np
from scipy import signal

def test(t):
    return t+1


def voltage_processor(dat):
    # V is the voltage data
    V = dat['V'].astype('float32')

    fs = dat["srate"]

    # high-pass filter above 50 Hz
    b, a = signal.butter(3, [50], btype='high', fs=fs)
    V = signal.filtfilt(b, a, V, 0)

    # compute smooth envelope of this signal = approx power
    V = np.abs(V)**2
    b, a = signal.butter(3, [10], btype='low', fs=fs)
    V = signal.filtfilt(b, a, V, 0)

    # normalize each channel so its mean power is 1
    V = V/V.mean(0)

    return V

def voltage_task_parser(V, stim_id, t_on):
    # average the broadband power across all trials per task
    nt, nchan = V.shape
    nstim = len(t_on)

    trange = np.arange(0, 2000)
    ts = t_on[:, np.newaxis] + trange
    V_epochs = np.reshape(V[ts, :], (nstim, 2000, nchan))

    '''
    modified: iterated over each task 
    '''
    task_parsed_Vs = {}
    for si in np.unique(stim_id):
        V_task = (V_epochs[stim_id == si]).mean(0)
        task_parsed_Vs[si] = V_task
    
    return task_parsed_Vs


def default_ECoG_processor(dat):
    V = voltage_processor(dat)
    processed_Vs = voltage_task_parser(V, dat["stim_id"], dat["t_on"])

    return processed_Vs
