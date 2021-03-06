import inspect
import os.path

import yaml

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from scipy.stats import linregress, norm
from keras.models import load_model
import matplotlib.pyplot as plt

from src.data_provider import prepare_data_inst_npz
from test_on_shift import main as main_shift


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


def generate_wgt_matrix(length=8):
    x = np.arange(length, dtype=np.float32)
    y = np.ones(shape=length, dtype=np.float32)
    a = np.stack((x, y), axis=0).T
    idn = np.identity(8, dtype=np.float32)
    wgt_matrix = a @ np.linalg.inv(a.T @ a) @ a.T - idn
    return wgt_matrix


def compute_loss(y_pred, y, group=8):
    wgt_matrix = generate_wgt_matrix(length=group)
    y_diff = y_pred - y
    y_wgt = np.reshape(
        np.tile(wgt_matrix, reps=[y.shape[0], 1]),
        newshape=(y.shape[0], group, group)
    )
    residuals = np.matmul(y_wgt, y_diff)
    total_loss = np.mean(residuals ** 2, axis=None)
    return total_loss


def compute_resolution(rms, slope, loss_hist):
    npz_content = np.load(loss_hist)
    hist = npz_content["hist"]
    slope_list = npz_content["slope_list"]
    reso_list = npz_content["reso_list"]
    # prepare mesh data for interpolation
    inter_inst = RectBivariateSpline(slope_list, reso_list, hist)
    init_reso = (reso_list[0] + reso_list[-1]) / 2
    # noinspection PyTypeChecker
    min_res = minimize(lambda p: (inter_inst(slope, p)[0, 0] - rms) ** 2, (init_reso,))
    reso = min_res.x[0]
    return reso


def main(config_file, base_model, loss_hist, shift_range, timestep):
    out_result_list, i_res_list, shift_start = main_shift(
        config_file=config_file,
        base_model=base_model,
        selected=0,
        shift_range=shift_range,
        timestep=timestep
    )

    # process shift results
    x = np.linspace(0, shift_range * timestep, shift_range, endpoint=False, dtype=np.float32)
    slope_list = [elem.slope for elem in i_res_list]
    intercept_list = [elem.intercept for elem in i_res_list]
    mean_slope = np.mean(slope_list)
    total_residual_squared = 0
    for i in range(out_result_list.shape[1]):
        y = out_result_list[:, i]
        i_int_init = intercept_list[i]
        # noinspection PyTypeChecker
        min_res = minimize(lambda p: np.sum((y - (p + mean_slope * x)) ** 2), (i_int_init,))
        i_int_opt = min_res.x[0]
        total_residual_squared += np.sum((y - (i_int_opt + mean_slope * x)) ** 2)

    mean_slope_abs = np.abs(mean_slope)
    mean_residual_squared = total_residual_squared / (out_result_list.shape[0] * out_result_list.shape[1])
    mean_residual_std = np.sqrt(mean_residual_squared)
    print("mean slope abs: %.3f" % mean_slope_abs)
    print("mean residual std.: %.3f" % mean_residual_std)

    shift_start = np.array(shift_start + [0, 0])
    shift_start_mean = np.mean(shift_start)
    shift_start_squared_mean = np.mean(shift_start ** 2)
    shift_start_std = np.sqrt(shift_start_squared_mean - shift_start_mean ** 2) * timestep
    print("randomness std.: %.3f" % shift_start_std)

    data_handler = get_data_handler(config_file=config_file)
    test_val_data_dict = data_handler.generate_test_val_dataset()
    inputs = test_val_data_dict["inputs"]
    targets = test_val_data_dict["targets"]
    labels = test_val_data_dict["labels"]
    targets_labels = np.zeros_like(labels)
    comb_inputs = np.concatenate((inputs, targets), axis=0)
    comb_labels = np.concatenate((labels, targets_labels), axis=0)

    out_result_list = np.zeros(shape=comb_inputs.shape[:2], dtype=np.float32)
    out_result_list = np.expand_dims(out_result_list, axis=-1)

    if isinstance(base_model, str):
        base_model = load_model(base_model, compile=False)

    for i in range(comb_inputs.shape[0]):
        out_result = base_model(comb_inputs[i, ...])
        out_result_list[i, ...] = out_result

    print("label shape:", comb_labels.shape)
    print("result shape:", out_result_list.shape)
    total_loss = compute_loss(y_pred=out_result_list, y=comb_labels)

    # use another method to compute the loss
    x_nn = np.arange(8)
    total_loss_alt = 0
    diff_list = np.zeros_like(out_result_list)
    for i in range(out_result_list.shape[0]):
        y_diff = out_result_list[i, :, 0] - comb_labels[i, :, 0]
        i_res = linregress(x_nn, y_diff)
        diff_list[i, :, 0] = y_diff - (i_res.intercept + i_res.slope * x_nn)
        total_loss_alt += np.mean((y_diff - (i_res.intercept + i_res.slope * x_nn)) ** 2, axis=None)
    total_loss_alt = total_loss_alt / out_result_list.shape[0]

    plt.figure()
    diff_list_flatten = np.reshape(diff_list, newshape=-1) * timestep
    plt.title("histogram of differences (%.3f)" % np.std(diff_list_flatten))
    plt.hist(diff_list_flatten, bins=200)
    plt.show()

    rms_test_data = np.sqrt(total_loss) * timestep
    rms_test_data_alt = np.sqrt(total_loss_alt) * timestep

    resolution = compute_resolution(
        rms=rms_test_data,
        slope=mean_slope_abs,
        loss_hist=loss_hist
    )
    model_unc = mean_residual_std * (1 / mean_slope_abs)
    data_unc = np.sqrt(resolution ** 2 - model_unc ** 2)

    print("--------RESULT---------")
    print(os.path.basename(config_file))
    print("slope: %.3f" % mean_slope_abs)
    print("rms of test data: %.3f" % rms_test_data)
    print("rms of test data (alternative): %.3f" % rms_test_data_alt)
    print("resolution: %.3f" % resolution)
    print("model uncertainty: %.3f" % model_unc)
    print("data uncertainty: %.3f" % data_unc)


if __name__ == "__main__":
    main(
        config_file="./conf/random_orig_l4_8ch_internal.yaml",
        base_model="./temp/random_orig_l4_8ch/compact_model/random_orig_l4_8ch-bind_seq_model",
        loss_hist="./temp/loss_hist/l4_8ch_hist.npz",
        shift_range=40,
        timestep=0.4
    )
