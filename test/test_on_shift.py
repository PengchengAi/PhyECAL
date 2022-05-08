import inspect
import yaml

from tensorflow.keras.models import load_model
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

from src.data_provider import prepare_data_inst_npz


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


def main(config_file, base_model, selected=0, shift_range=20, timestep=0.4, debug=False):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    sig_cfg = cfg["signal_ch"]
    start = sig_cfg["sample_start"][0]
    sample_every = sig_cfg["sample_every"]
    sample_points = sig_cfg["sample_points"]
    sample_length = sample_every * sample_points
    channel_count = sig_cfg["channel_count"]
    baseline_range = [0, start]

    raw_npz_file = cfg["data_raw_file"]["bind"][0]
    raw_content = np.load(raw_npz_file)
    selected_raw = raw_content["data"][selected, ...]

    for j in range(channel_count):
        j_baseline = np.mean(selected_raw[j, baseline_range[0]:baseline_range[1]], axis=None)
        selected_raw[j, :] = selected_raw[j, :] - j_baseline

    data_handler = get_data_handler(config_file=config_file)
    wave_scale, wave_shift = data_handler.get_norm_transform(item="decoder")

    if isinstance(base_model, str):
        base_model = load_model(base_model, compile=False)

    shift_data_set = np.zeros(shape=(shift_range, channel_count, sample_points), dtype=np.float32)
    for i in range(shift_range):
        i_start = start + i - shift_range // 2
        shift_data_set[i, ...] = selected_raw[:, i_start:i_start + sample_length:sample_every] * wave_scale + wave_shift
        if debug:
            plt.figure()
            for j in range(channel_count):
                plt.plot(shift_data_set[i, j, :])
            plt.show()
    shift_data_set = np.expand_dims(shift_data_set, axis=-1)

    print(shift_data_set.shape)

    total_samples = shift_data_set.shape[0]
    out_result_list = np.zeros(shape=shift_data_set.shape[:2], dtype=np.float32)

    for i in range(total_samples):
        out_result = base_model(shift_data_set[i, ...])
        out_result_list[i, ...] = np.squeeze(out_result, axis=-1)

    print(out_result_list.shape)

    plt.figure()
    plt.title("Predict shift waveform")
    x = np.linspace(0, shift_range * timestep, shift_range, endpoint=False, dtype=np.float32)
    out_result_list = out_result_list * timestep

    i_res_list = []
    plt.xlabel("shifted time (ns)")
    plt.ylabel("predicted time (ns)")
    for i in range(channel_count):
        i_res = linregress(x, out_result_list[:, i])
        i_res_list.append(i_res)
        slope = i_res.slope
        intercept = i_res.intercept
        residuals = out_result_list[:, i] - (intercept + slope * x)
        residuals_std = np.std(residuals)
        plt.plot(x, intercept + slope * x, label="ch_%d fit" % i, linestyle="-.")
        plt.plot(x, out_result_list[:, i], label="ch_%d (%.2f)" % (i, float(residuals_std)))
        print("channel %d: intercept %.2f, slope %.2f, std. of residuals %.2f" % (
            i, intercept, slope, float(residuals_std)))
    plt.legend()
    plt.show()

    return out_result_list, i_res_list, sig_cfg["shift_start"]


if __name__ == "__main__":
    main(
        config_file="./conf/random_orig_l4_8ch_internal.yaml",
        base_model="./temp/random_orig_l4_8ch/compact_model/random_orig_l4_8ch-bind_seq_model",
        selected=0,
        shift_range=40
    )
