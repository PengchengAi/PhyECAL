import os
import inspect
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm

from src.data_provider import prepare_data_inst_npz


def polynomial_fit(curve, timestep=0.4, origin=0., baseline=0., order=3, p0=(2.0, 0.0, -1.e-3, 3.e3, -1.e-4),
                   thresh=0., x0=10., visualize=False):
    def poly_func(x, start, *coe):
        base = coe[0]

        def passed_func(value):
            passed = base
            for i in range(1, order+1):
                passed += coe[i] * (value - start) ** i
            return passed

        result = np.piecewise(x, [x < start, x >= start], [base, passed_func])
        return result

    assert order + 2 == len(p0), "The initial parameters do not match the order."
    curve_rebase = curve - baseline
    xdata = np.linspace(origin, origin + timestep * len(curve), len(curve), endpoint=False, dtype=np.float64)
    popt, pcov = curve_fit(poly_func, xdata, curve_rebase, p0=p0)
    x_solve = fsolve(lambda x: poly_func(x, *popt) - thresh, x0=np.array(x0))
    if visualize:
        print("optimized parameters:", popt)
        print("coefficients:", pcov)
        plt.figure()
        plt.plot(xdata, curve_rebase, "g.-", label="original signal (rebased)")
        plt.plot(xdata, poly_func(xdata, *popt), "b--", label="fitting function")
        plt.axvline(x=x_solve, linestyle="--", label="solved x")
        plt.axhline(y=thresh, linestyle="-.", label="threshold")
        plt.xlabel("time (ns)")
        plt.ylabel("normalized amplitude")
        plt.title("Use polynomial to fit samples")
        plt.legend()
        plt.show()
    return popt, x_solve


def get_data_handler(config_file, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]

    if "upd_kwargs" in base_cfg:
        kwargs.update(base_cfg["upd_kwargs"])

    kwargs["config_file"] = config_file
    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_inst_npz)[0]}

    data_handler = prepare_data_inst_npz(**data_kwargs)

    return data_handler


def ta_fit_func(a, c1, c2, c3):
    return a + c1 / np.sqrt(a) + c2 / a + c3 * a


def time_amp_correction(data, amp, p0=(0.1, 0.1, 0.1)):
    popt, pcov = curve_fit(ta_fit_func, amp, data, p0=p0)
    new_data = data - ta_fit_func(amp, *popt)
    return new_data, popt


def plot_data(data, amp, correct_popt=None, title=None):
    plt.figure()
    plt.xlabel("normalized amplitude")
    plt.ylabel("time (ns)")
    _, xedges, _, _ = plt.hist2d(
        amp,
        data,
        range=[[0.2, 1.], [-40., 40.]],
        bins=40,
        cmap=plt.cm.get_cmap("jet")
    )
    if correct_popt is not None:
        plt.plot(xedges, ta_fit_func(xedges, *correct_popt), "r-")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()

    plt.figure()
    d_mean = np.nanmean(data, axis=None)
    d_std = np.nanstd(data, axis=None)
    plt.xlabel("time (ns)")
    plt.ylabel("probability density")
    _, bins, _ = plt.hist(data, bins=40, density=True)
    plt.plot(bins, norm.pdf(bins, loc=d_mean, scale=d_std), "r-.")
    if title is not None:
        plt.title("%s\n std.: %.3f" % (title, float(d_std)))
    else:
        plt.title("std.: %.3f" % float(d_std))
    plt.tight_layout()
    plt.show()


def task_sub_channel_0_1(result_list):
    data = (result_list[:, 1, 0] - result_list[:, 0, 0]) / np.sqrt(2)
    amp_0 = result_list[:, 0, 1]
    amp_1 = result_list[:, 1, 1]

    data_cor_0, popt_0 = time_amp_correction(data=data, amp=amp_0)
    data_cor_01, popt_01 = time_amp_correction(data=data_cor_0, amp=amp_1)

    # initial fit
    plot_data(
        data=data,
        amp=amp_0,
        correct_popt=popt_0,
        title="original"
    )

    # time-amplitude correction for channel 0
    plot_data(
        data=data_cor_0,
        amp=amp_1,
        correct_popt=popt_01,
        title="amp_0 corrected"
    )

    # time-amplitude correction for channel 0 and 1
    plot_data(
        data=data_cor_01,
        amp=amp_1,
        title="amp_0 and amp_1 corrected"
    )


def task_sub_channel_0_1_2(result_list):
    data = (result_list[:, 0, 0] + result_list[:, 2, 0] - 2 * result_list[:, 1, 0]) / np.sqrt(6)
    amp_0 = result_list[:, 0, 1]
    amp_1 = result_list[:, 1, 1]
    amp_2 = result_list[:, 2, 1]

    data_cor_0, popt_0 = time_amp_correction(data=data, amp=amp_0)
    data_cor_01, popt_01 = time_amp_correction(data=data_cor_0, amp=amp_1)
    data_cor_012, popt_012 = time_amp_correction(data=data_cor_01, amp=amp_2)

    # initial fit
    plot_data(
        data=data,
        amp=amp_0,
        correct_popt=popt_0,
        title="original"
    )

    # time-amplitude correction for channel 0
    plot_data(
        data=data_cor_0,
        amp=amp_1,
        correct_popt=popt_01,
        title="amp_0 corrected"
    )

    # time-amplitude correction for channel 0 and 1
    plot_data(
        data=data_cor_01,
        amp=amp_2,
        correct_popt=popt_012,
        title="amp_0 and amp_1 corrected"
    )

    # time-amplitude correction for channel 0, 1 and 2
    plot_data(
        data=data_cor_012,
        amp=amp_2,
        title="amp_0, amp_1 and amp_2 corrected"
    )


def task_sub_all_three_channels(result_list):
    npz_sel_ind_content = np.load("./temp/test_on_linear_cut_selected_index.npz")
    sel_index = npz_sel_ind_content["selected_index"]
    assert result_list.shape[0] == len(sel_index), "Length mismatches."
    result_list = result_list[sel_index]

    data_gather = []
    for i in range(1, 7, 1):
        data = (result_list[:, i-1, 0] + result_list[:, i+1, 0] - 2 * result_list[:, i, 0]) / np.sqrt(6)
        amp_l = result_list[:, i-1, 1]
        amp_c = result_list[:, i, 1]
        amp_h = result_list[:, i+1, 1]

        data_cor_l, _ = time_amp_correction(data=data, amp=amp_l)
        data_cor_lc, _ = time_amp_correction(data=data_cor_l, amp=amp_c)
        data_cor_lch, _ = time_amp_correction(data=data_cor_lc, amp=amp_h)
        data_gather.append(data_cor_lch)

    data_gather = np.reshape(data_gather, newshape=-1)
    data_gather_f = data_gather[np.logical_and(data_gather > -2, data_gather < 2)]

    plt.figure()
    title = "all three channels"
    d_mean = np.nanmean(data_gather_f, axis=None)
    d_std = np.nanstd(data_gather_f, axis=None)
    plt.xlabel("time (ns)")
    plt.ylabel("probability density")
    _, bins, _ = plt.hist(data_gather, bins=80, range=[-5., 5.], density=False)
    plt.plot(bins, norm.pdf(bins, loc=d_mean, scale=d_std) * len(data_gather_f) * (bins[1] - bins[0]), "r-.")
    if title is not None:
        plt.title("%s\n std.: %.3f" % (title, float(d_std)))
    else:
        plt.title("std.: %.3f" % float(d_std))
    plt.tight_layout()
    plt.show()


def main(config_file, thresh, result_path=None, run_test=True, sel_range=(-70, -20), visualize=False, debug=False):
    data_handler = get_data_handler(config_file=config_file)

    test_data_dict = data_handler.generate_test_val_dataset()
    targets = test_data_dict["targets"]
    targets = np.squeeze(targets, axis=-1)
    assert len(targets.shape) == 3, "The shape of data is not expected."
    print(targets.shape)
    amount, channel_count, sample_points = targets.shape

    if result_path is None or not os.path.exists(result_path) or run_test:
        if not run_test:
            print("Test still runs because result path is None or does not exist.")
        result_list = np.zeros(shape=(amount, channel_count, 2), dtype=np.float32)

        for i in range(amount):
            for j in range(channel_count):
                wave = targets[i, j, :]
                peak_ind = np.argmax(wave, axis=None)
                if debug:
                    print("peak value:", wave[peak_ind])
                # valid should be in the range
                if peak_ind + sel_range[0] < 0:
                    result_list[i, j, :] = np.nan
                    continue
                try:
                    wave_for_fit = wave[peak_ind + sel_range[0]:peak_ind + sel_range[1]]
                except Warning:
                    print("Something is wrong with fitting.")
                    result_list[i, j, :] = np.nan
                    continue
                _, x_solve = polynomial_fit(wave_for_fit, thresh=thresh, visualize=visualize)
                # noinspection PyTypeChecker
                result_list[i, j, 0] = float(peak_ind + sel_range[0]) * 0.4 + x_solve
                result_list[i, j, 1] = wave[peak_ind]

        if result_path is not None:
            np.savez(
                result_path,
                result=result_list
            )
    else:
        npz_content = np.load(result_path)
        result_list = npz_content["result"]

    task_sub_channel_0_1(result_list=result_list)
    task_sub_channel_0_1_2(result_list=result_list)
    task_sub_all_three_channels(result_list=result_list)


if __name__ == "__main__":
    main(
        config_file="./conf/random_orig_l4_8ch_internal.yaml",
        thresh=-0.05,
        result_path="./temp/test_fit_method_result_cache.npz",
        run_test=False,
        visualize=False
    )
