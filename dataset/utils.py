import pandas as pd
import os
import glob
import re
from scipy import io
from tqdm.auto import tqdm

from evaluate import postprocess


def getFold(path):
    ret = {}
    files = glob.glob(path + os.sep + "fold" + os.sep + "*.mat")
    for f in files:
        i = int(f[-5])
        temp = io.loadmat(f)[f"fold{i}"][0]  # subject_idx of fold(i + 1)
        for idx in temp:
            ret[idx] = i
    return ret


def getRecord(base_path):
    record = {"video": [], "path": [], "idx": [], "beat_num": [],
              "fold": [], "task": [], "source": []}
    folds = getFold(base_path)
    dirs = glob.glob(base_path + os.sep + "data" + os.sep + "*")
    bar = tqdm(range(len(dirs)))
    for pi in dirs:
        p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
        tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
        for ti in tasks:
            if not re.findall("v(\d-\d)", ti):
                t_idx = re.findall("v(\d)", ti)[0]
            else:
                t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
            sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
            for si in sources:
                s_idx = re.findall("source(\d)", si)[0]  # source_i
                video = f"p{p_idx}_v{t_idx}_source{s_idx}"
                clips = glob.glob(si + os.sep + "cvd_cache" + os.sep + "*")
                for ci in clips:
                    idx = ci.split(os.sep)[-1]
                    record["video"].append(video)
                    record["path"].append(ci)
                    record["idx"].append(idx)
                    record["fold"].append(folds[int(p_idx)])
                    record["task"].append(int(t_idx[0]))
                    record["source"].append(int(s_idx))

                    bvp = io.loadmat(ci + "/bvp.mat")["bvp"].reshape(-1)
                    Fs = io.loadmat(ci + "/fps.mat")["fps"].reshape(-1)
                    hr = postprocess.fft_physiology(signal=bvp, Fs=Fs, diff=False, 
                                                    detrend_flag=True)[0]
                    record["beat_num"].append(float(hr / 60 * 300 / Fs))  # clip_len = 300
        bar.update(1)
    return pd.DataFrame(record)


def appendAug(base_path, record: pd.DataFrame):
    folds = getFold(base_path)
    dirs = glob.glob(base_path + os.sep + "data" + os.sep + "*")
    bar = tqdm(range(len(dirs)))
    for pi in dirs:
        p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
        tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
        for ti in tasks:
            if not re.findall("v(\d-\d)", ti):
                t_idx = re.findall("v(\d)", ti)[0]
            else:
                t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
            sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
            for si in sources:
                s_idx = re.findall("source(\d)", si)[0]  # source_i
                video = f"p{p_idx}_v{t_idx}_source{s_idx}"

                ups = glob.glob(si + os.sep + "up_samples" + os.sep + "*")
                for ci in ups:
                    idx = ci.split(os.sep)[-1]
                    bvp = io.loadmat(ci + "/bvp.mat")["bvp"].reshape(-1)
                    Fs = io.loadmat(ci + "/fps.mat")["fps"].reshape(-1)
                    hr = postprocess.fft_physiology(signal=bvp, Fs=Fs, diff=False, 
                                                    detrend_flag=True)[0]
                    info = {"video": [video], "path": [ci], "idx": [idx], 
                            "beat_num": [float(hr / 60 * 300 / Fs)], "fold": [folds[int(p_idx)]], 
                            "task": [int(t_idx[0])], "source": [int(s_idx)]}
                    # record = record.append(info, ignore_index=True)
                    record = pd.concat([record, pd.DataFrame(info)], ignore_index=True)

                downs = glob.glob(si + os.sep + "down_samples" + os.sep + "*")
                for ci in downs:
                    idx = ci.split(os.sep)[-1]
                    bvp = io.loadmat(ci + "/bvp.mat")["bvp"].reshape(-1)
                    Fs = io.loadmat(ci + "/fps.mat")["fps"].reshape(-1)
                    hr = postprocess.fft_physiology(signal=bvp, Fs=Fs, diff=False, 
                                                    detrend_flag=True)[0]
                    info = {"video": [video], "path": [ci], "idx": [idx], 
                            "beat_num": [float(hr / 60 * 300 / Fs)], "fold": [folds[int(p_idx)]], 
                            "task": [int(t_idx[0])], "source": [int(s_idx)]}
                    # record = record.append(info, ignore_index=True)
                    record = pd.concat([record, pd.DataFrame(info)], ignore_index=True)
        
        bar.update(1)
    return record
