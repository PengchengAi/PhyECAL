import os

import numpy as np
import matplotlib.pyplot as plt


def show(sel_data):
    ch_cnt, pt_cnt = sel_data.shape
    plt.figure()
    x = np.linspace(0., pt_cnt * 0.4, pt_cnt, endpoint=False, dtype=np.float64)
    for i in range(ch_cnt):
        plt.plot(x, sel_data[i], label="wave_%d" % i)
    plt.legend()
    plt.show()


def main(data_dir, save_dir=None, plot=True):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "wave_raw.npz")
        if os.path.exists(save_path):
            raise Exception("Save file already exists.")
    else:
        save_path = None

    data_arr = []
    for i in range(8):
        file_path = os.path.join(data_dir, "wave_%d.dat" % i)
        data = np.fromfile(file=file_path, dtype=np.float32)
        data_arr.append(data)
    data_arr = np.array(data_arr, dtype=np.float32)
    assert len(data_arr.shape) == 2, "Shape error."
    ch_cnt, tot_pt_cnt = data_arr.shape
    cnt = tot_pt_cnt // 1024
    pass_cnt = 0
    if save_path is not None:
        sel_data_arr = []
        sel_ind_arr = []
    else:
        sel_data_arr = None
        sel_ind_arr = None
    for i in range(cnt):
        sel_data = data_arr[:, 1024*i:1024*i+1000]
        valid_by_amp = True
        for j in range(ch_cnt):
            sel_ch = sel_data[j]
            sel_ch_min = np.min(sel_ch)
            sel_ch_max = np.max(sel_ch)
            if not (80. <= sel_ch_max - sel_ch_min <= 400.):
                valid_by_amp = False
        if valid_by_amp:
            pass_cnt += 1
            if save_path is not None:
                sel_data_arr.append(sel_data)
                sel_ind_arr.append(i)
            if plot:
                show(sel_data)

    print("Total: %d. Passed: %d." % (cnt, pass_cnt))

    if save_path is not None:
        sel_data_arr = np.array(sel_data_arr, dtype=np.float32)
        assert sel_data_arr.shape == (pass_cnt, ch_cnt, 1000), "Shape error."
        sel_ind_arr = np.array(sel_ind_arr, dtype=int)
        np.savez(
            save_path,
            index=sel_ind_arr,
            data=sel_data_arr,
            count=pass_cnt,
            source=data_dir
        )

        print("%d data saved to: %s" % (pass_cnt, save_path))


if __name__ == "__main__":
    main(
        data_dir="F:/Data/ECAL8ch/210929_01",
        save_dir="F:/Data/ECAL8ch/210929_01_npz",
        plot=False
    )
