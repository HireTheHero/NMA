from copy import deepcopy


import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import rankdata
from sklearn.cluster import KMeans


def test(t):
    return t + 1


def voltage_processor(dat):
    # V is the voltage data
    V = dat["V"].astype("float32")

    fs = dat["srate"]

    # high-pass filter above 50 Hz
    b, a = signal.butter(3, [50], btype="high", fs=fs)
    V = signal.filtfilt(b, a, V, 0)

    # compute smooth envelope of this signal = approx power
    V = np.abs(V) ** 2
    b, a = signal.butter(3, [10], btype="low", fs=fs)
    V = signal.filtfilt(b, a, V, 0)

    # normalize each channel so its mean power is 1
    V = V / V.mean(0)

    return V


def voltage_task_parser(V, stim_id, t_on):
    # average the broadband power across all trials per task
    nt, nchan = V.shape
    nstim = len(t_on)

    trange = np.arange(0, 2000)
    ts = t_on[:, np.newaxis] + trange
    V_epochs = np.reshape(V[ts, :], (nstim, 2000, nchan))

    """
    modified: iterated over each task 
    """
    task_parsed_Vs = {}
    for si in np.unique(stim_id):
        V_task = (V_epochs[stim_id == si]).mean(0)
        task_parsed_Vs[f"stimuli_{si}"] = V_task

    return task_parsed_Vs


def default_ECoG_processor(dat):
    V = voltage_processor(dat)
    processed_Vs = voltage_task_parser(V, dat["stim_id"], dat["t_on"])

    return processed_Vs


def same_max_electrode_concatter(locs, nums_electrodes, cluster_by_all=False):
    # create map out of locations
    ## pick 64-electrode subjects and others
    subjects = np.array(list(nums_electrodes.keys()))
    subj_64 = subjects[
        np.array(list(nums_electrodes.values()))
        == np.max(list(nums_electrodes.values()))
    ]
    ## concat 64-electrode positions
    loc_64 = locs[subj_64[0]]
    if cluster_by_all:
        for subj in subj_64[1:]:
            loc = locs[subj]
            loc_64 = np.vstack((loc_64, loc))
    else:
        pass
    return loc_64


def electrode_ranker(locs):
    locs_out = {}
    for subj in locs.keys():
        loc = locs[subj]
        locs_out[subj] = rankdata(loc).reshape(loc.shape)
    return locs_out


def electrode_mapper(locs, nums_electrodes, random_state=19871221, relative=True):
    if relative:
        locs_input = electrode_ranker(locs)
    else:
        locs_input = deepcopy(locs)
    loc_64 = same_max_electrode_concatter(locs_input, nums_electrodes)
    print(f"Number of electrodes for indexing: {len(loc_64)}")
    model = KMeans(
        n_clusters=np.max(list(nums_electrodes.values())), random_state=random_state
    )
    model = model.fit(loc_64)
    electrode_nums = {}
    for subj in locs_input.keys():
        electrode_num = model.predict(locs_input[subj])
        electrode_nums[subj] = electrode_num
    return electrode_nums


def category_extractor(dat):
    subjects = list(dat.keys())
    experiments = list(dat[subjects[0]].keys())
    stimuli = list(dat[subjects[0]][experiments[0]].keys())

    return {"subjects": subjects, "experiments": experiments, "stimuli": stimuli}


def dat_2_df_converter(dat, categories):
    out = []
    for subject in categories["subjects"]:
        for experiment in categories["experiments"]:
            for stimulus in categories["stimuli"]:
                result = dat[subject][experiment][stimulus]
                stim_len, _ = result.shape
                times = np.array(range(stim_len))
                sbj_dat_reshaped = times
                sbj_dat_reshaped = np.vstack(
                    (sbj_dat_reshaped, np.repeat(subject, stim_len))
                )
                sbj_dat_reshaped = np.vstack(
                    (sbj_dat_reshaped, np.repeat(experiment, stim_len))
                )
                sbj_dat_reshaped = np.vstack(
                    (sbj_dat_reshaped, np.repeat(stimulus, stim_len))
                )
                sbj_dat_reshaped = np.vstack((sbj_dat_reshaped, result.T))
                out.append(pd.DataFrame(sbj_dat_reshaped.T))
    out = pd.concat(out)
    return out


def dat_df_generator(dat, meta_cols=["t", "subject", "experiment", "stimulus"]):
    categories = category_extractor(dat)
    dat_pd_list = dat_2_df_converter(dat, categories)
    electrode_indices = np.array(range(len(dat_pd_list.columns) - len(meta_cols)))
    electrode_cols = [f"electrode_{idx}" for idx in electrode_indices]
    dat_pd_list.columns = meta_cols + electrode_cols
    dat_pd_list["t"] = dat_pd_list["t"].astype(int)
    dat_pd_list[electrode_cols] = dat_pd_list[electrode_cols].astype(float)

    return dat_pd_list


def convert_to_batches(data, batch_num=5):
    if type(data) in [list, pd.core.indexes.base.Index]:
        out = [data[i : i + batch_num] for i in range(0, len(data), batch_num)]
    else:
        # do nothing
        out = deepcopy(data)
    return out
