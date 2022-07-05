import os

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import norm
import matplotlib.pyplot as plt


class MaxConstantFractionDiscrimination(object):
    def __init__(self, ratio=None, baseline=0.0, timestep=0.01):
        """
        Initialize a maximum (digital) constant fraction discriminator (dCFD).

        :param ratio: The discount ratio (usually between 0 and 1) of the original signal in dCFD.
        :param baseline: The baseline (pedestal) used by the dCFD algorithm.
        :param timestep: The interval between two samples of the incoming waveform (in ns).
        """
        super(MaxConstantFractionDiscrimination, self).__init__()
        self._ratio = ratio
        self._base = baseline
        self._ts = timestep

    def apply(self, input_series: np.ndarray, ratio=None, baseline=None, origin=None, timestep=None, visualize=False):
        """
        The interface function of the dCFD instance.

        :param input_series: A numpy ndarray representing the input waveform.
        :param ratio: The discount ratio (usually between 0 and 1) of the original signal in dCFD (override).
        :param baseline: The baseline (pedestal) used by the dCFD algorithm (override).
        :param origin: The ground-truth time of the first sample point (in ns).
        :param timestep: The interval between two samples of the incoming waveform (in ns, override).
        :param visualize: Switch on visualization to debug the result from the dCFD algorithm.
        :return: The predicted time, or None if the algorithm fails.
        """
        assert ratio is not None or self._ratio is not None, "Ratio is required."
        assert baseline is not None or self._base is not None, "Baseline is required."
        assert timestep is not None or self._ts is not None, "Timestep is required."
        if ratio is None:
            ratio = self._ratio
        if baseline is None:
            baseline = self._base
        if timestep is None:
            timestep = self._ts
        cfd_series = input_series - baseline
        i = np.argmax(cfd_series)
        if i == 0:
            print("The peak of the waveform comes too early.")
            return None
        else:
            cfd_level = cfd_series[i] * ratio  # cfd level is smaller than the maximum value
            j = 0
            for j in range(0, i+1):
                if cfd_series[j] >= cfd_level:
                    break
            if j == 0:
                print("The ratio is too small for this waveform.")
                return None
            val_before = cfd_series[j - 1]
            val_after = cfd_series[j]
            val_int = cfd_level
            if np.abs(val_after - val_before) <= np.finfo(np.float64).eps:
                time_adj = timestep / 2
            else:
                time_adj = timestep * (val_int - val_before) / (val_after - val_before)
            result = (j - 1) * timestep + time_adj
            if origin is not None:
                result = origin + result
            else:
                origin = 0.

            if visualize:
                x = np.linspace(origin, origin + len(input_series) * timestep, len(input_series), endpoint=False)
                plt.figure()
                plt.plot(x, input_series, c="b", label="original signal")
                plt.axvline(x=result, c="r", label="predicted time")
                plt.axhline(y=cfd_level+baseline, c="y", label="CFD level")
                plt.plot((x[j-1], x[j]), (val_before+baseline, val_after+baseline), "go-", label="linear approx.")
                plt.xlabel("time (ns)")
                plt.ylabel("amplitude (V)")
                plt.title("%.4e" % result)
                plt.legend()
                plt.show()

        return result


class IntConstantFractionDiscrimination(MaxConstantFractionDiscrimination):
    def __init__(self, ratio=None, baseline=0.0, int_points=4, win_order=28, win_beta=8.6, timestep=0.01):
        """
        Initialize a maximum (digital) constant fraction discriminator (dCFD) with interpolation.

        :param ratio: The discount ratio (usually between 0 and 1) of the original signal in dCFD.
        :param baseline: The baseline (pedestal) used by the dCFD algorithm.
        :param int_points: One in int_points is from input waveform, and others are zero.
        :param win_order: The order of the FIR window.
        :param win_beta: The beta of the FIR window.
        :param timestep: The interval between two samples of the incoming waveform (in ns).
        """
        super(IntConstantFractionDiscrimination, self).__init__(ratio, baseline, timestep)
        self._ipts = int_points
        self._order = win_order
        self._beta = win_beta

    def apply(self, input_series: np.ndarray, ratio=None, baseline=None, int_points=None, win_order=None, win_beta=None,
              origin=None, timestep=None, visualize=False):
        """
        The interface function of the dCFD instance with interpolation.

        :param input_series: A numpy ndarray representing the input waveform.
        :param ratio: The discount ratio (usually between 0 and 1) of the original signal in dCFD (override).
        :param baseline: The baseline (pedestal) used by the dCFD algorithm (override).
        :param int_points: One in int_points is from input waveform, and others are zero (override).
        :param win_order: The order of the FIR window (override).
        :param win_beta: The beta of the FIR window (override).
        :param origin: The ground-truth time of the first sample point (in ns).
        :param timestep: The interval between two samples of the incoming waveform (in ns, override).
        :param visualize: Switch on visualization to debug the result from the dCFD algorithm.
        :return: The predicted time, or None if the algorithm fails.
        """
        assert baseline is not None or self._base is not None, "Baseline is required."
        assert int_points is not None or self._ipts is not None, "Number of interpolating points is required."
        assert win_order is not None or self._order is not None, "Number of filter order is required."
        assert win_beta is not None or self._beta is not None, "Beta parameter is required."
        assert timestep is not None or self._ts is not None, "Timestep is required."
        if baseline is None:
            baseline = self._base
        if int_points is None:
            int_points = self._ipts
        if win_order is None:
            win_order = self._order
        if win_beta is None:
            win_beta = self._beta
        if timestep is None:
            timestep = self._ts

        window = signal.windows.kaiser(M=win_order, beta=win_beta)
        int_series = np.ones(shape=int_points*len(input_series), dtype=np.float64) * baseline
        int_series[0::int_points] = input_series
        zi = signal.lfiltic(b=window, a=[1], y=[], x=[baseline]*(len(window)-1))
        flt_series, _ = signal.lfilter(b=window, a=[1], x=int_series, zi=zi)
        flt_baseline = np.sum(window * np.ones(shape=len(window), dtype=np.float64) * baseline)

        result = super(IntConstantFractionDiscrimination, self).apply(input_series=flt_series,
                                                                      ratio=ratio,
                                                                      baseline=flt_baseline,
                                                                      origin=origin,
                                                                      timestep=timestep/float(int_points),
                                                                      visualize=visualize)
        return result


def extract_data(path, verbose=0):
    # extract data
    df = pd.read_csv(path)
    if verbose >= 1:
        print(df.info())
    df_vld = df.iloc[1:]
    df_vld = df_vld.convert_dtypes()  # an error will occur without this statement
    if verbose >= 2:
        print(df_vld.info())
    df_vld[:] = df_vld[:].astype(np.int32)
    if verbose >= 2:
        print(df_vld.info())
    df_vld = df_vld.loc[df_vld["mon_ckpt_inst/valid_flag"] == 1]
    if verbose >= 2:
        print(df_vld.info())
    adc_data = df_vld["mon_ckpt_inst/mem_dout[11:0]"].to_numpy()
    if verbose >= 2:
        print(adc_data)
    adc_data = np.reshape(adc_data, newshape=(-1, 2, 32))
    return adc_data


def single_file_routine(path, verbose=0):
    adc_data = extract_data(path=path, verbose=verbose)
    assert len(adc_data.shape) == 3 and adc_data.shape[1] == 2, "Shape mismatches."
    algo_inst = IntConstantFractionDiscrimination(ratio=0.5, timestep=1000./125.)
    # apply algorithms
    time_diff = []
    for i in range(adc_data.shape[0]):
        event_adc_data = adc_data[i, :, :]
        result = np.zeros(shape=2, dtype=np.float64)
        for j in range(2):
            ch_adc_data = event_adc_data[j, :]
            result[j] = algo_inst.apply(input_series=ch_adc_data, baseline=ch_adc_data[0], visualize=(verbose >= 2))
        time_diff.append(result[1] - result[0])
    # compute statistics
    time_diff = np.array(time_diff)
    mean = np.mean(time_diff)
    std = np.std(time_diff)
    # plot results
    if verbose >= 1:
        plt.figure()
        plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(time_diff), float(mean), float(std)))
        _, bins, _ = plt.hist(time_diff, density=True)
        plt.plot(bins, norm.pdf(bins, loc=mean, scale=std), "r--")
        plt.show()
    return time_diff


def multi_file_routine(dirname, file_cnt=10, file_prefix="iladata", verbose=0):
    time_diff_col = []
    for i in range(file_cnt):
        filename = file_prefix + str(i+1) + ".csv"
        file_path = os.path.join(dirname, filename)
        time_diff = single_file_routine(path=file_path, verbose=verbose)
        time_diff_col.append(time_diff)
    time_diff_col = np.reshape(np.array(time_diff_col), newshape=-1)
    # compute statistics
    time_diff_col = np.array(time_diff_col)
    mean = np.mean(time_diff_col)
    std = np.std(time_diff_col)
    # plot results
    plt.figure()
    plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(time_diff_col), float(mean), float(std)))
    _, bins, _ = plt.hist(time_diff_col, density=True)
    x = np.linspace(bins[0], bins[-1], 101, endpoint=True)
    plt.plot(x, norm.pdf(x, loc=mean, scale=std), "r--")
    plt.show()


if __name__ == "__main__":
    # single_file_routine(
    #     path="D:\\FPGAPrj\\nn_daq_trigger\\saved_data\\20220704\\iladata1.csv",
    #     verbose=1
    # )
    multi_file_routine(
        dirname="D:\\FPGAPrj\\nn_daq_trigger\\saved_data\\20220704",
        file_cnt=10,
        file_prefix="iladata",
        verbose=1
    )
