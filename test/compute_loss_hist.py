import os

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def compute(shift_start, slope_list, reso_list, sample_cnt=1000, channels=8, timestep=0.4, with_target=True):
    res_hist = np.zeros(shape=(len(slope_list), len(reso_list)))
    x = np.arange(0, channels)

    if with_target:
        data = np.random.normal(loc=0., scale=1., size=(sample_cnt, channels))
        total_loss = 0.
        for s in range(sample_cnt):
            s_res = linregress(x, data[s, :])
            total_loss += np.mean((data[s, :] - (s_res.intercept + s_res.slope * x)) ** 2, axis=None)
        unit_loss = total_loss / sample_cnt
    else:
        unit_loss = 0.

    for i, slope in enumerate(slope_list):
        for j, reso in enumerate(reso_list):
            total_loss = 0.
            remaining_samples = sample_cnt
            if with_target:
                total_loss += unit_loss * (slope ** 2) * (reso ** 2) * (sample_cnt // 2)
                remaining_samples = remaining_samples - (sample_cnt // 2)
            data_1 = np.random.normal(loc=0., scale=slope * reso, size=(remaining_samples, channels))
            data_2 = np.random.choice(
                shift_start,
                size=(remaining_samples, channels), replace=True) * timestep * (slope - 1.)
            data = data_1 + data_2
            for s in range(remaining_samples):
                s_res = linregress(x, data[s, :])
                total_loss += np.mean((data[s, :] - (s_res.intercept + s_res.slope * x)) ** 2, axis=None)
            mean_loss = total_loss / sample_cnt
            res_hist[i, j] = np.sqrt(mean_loss)
        print("finish No. %d slope: %.3f" % (i, slope))

    return res_hist


def main(save_dir, shift_start_range, plot=True):
    slope_list = np.linspace(0.01, 0.99, 20, endpoint=True)
    reso_list = np.linspace(0.1, 1., 24, endpoint=True)
    for i in shift_start_range:
        shift_start = [-i, i]
        print("computation for shift start: %d" % i)
        res_hist = compute(
            shift_start=shift_start,
            slope_list=slope_list,
            reso_list=reso_list
        )
        save_path = os.path.join(save_dir, "l%d_8ch_hist.npz" % i)
        np.savez(
            save_path,
            hist=res_hist,
            slope_list=slope_list,
            reso_list=reso_list
        )
        if plot:
            plt.figure()
            plt.title("test rms (ns) vs. resolution & slope at %d random shift" % i)
            plt.imshow(res_hist, cmap=plt.cm.get_cmap("jet"))
            plt.xlabel("resolution (ns)")
            x = np.arange(24)
            plt.xticks(x[::4], ["%.3f" % elem for elem in reso_list[::4]])
            plt.ylabel("slope")
            y = np.arange(20)
            plt.yticks(y[::4], ["%.3f" % elem for elem in slope_list[::4]])
            plt.colorbar()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main(
        save_dir="./temp/loss_hist/",
        shift_start_range=range(1, 5),
        plot=False
    )
