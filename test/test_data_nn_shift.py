import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress, norm

from src.build_model import build_seq_model, build_seq_model_base, ToyPhyBindModel, compile_model
from src.quan_aux import quantize_scope, quantize_apply
from conf.net_config import AVAILABLE_CONFIGS


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


def extract_data_multi(dirname, file_cnt=10, file_prefix="iladata", verbose=0):
    adc_data_col = []
    for i in range(file_cnt):
        filename = file_prefix + str(i+1) + ".csv"
        file_path = os.path.join(dirname, filename)
        adc_data = extract_data(path=file_path, verbose=verbose)
        adc_data_col.append(adc_data)
    adc_data_col = np.concatenate(adc_data_col, axis=0)
    return adc_data_col


def apply_nn(training_data, test_data, net_cfg_key="default_in32", net_compile_key="adam", timestep=8., verbose=0):
    # prepare model
    model_cfg = AVAILABLE_CONFIGS[net_cfg_key]
    base_model = build_seq_model_base(cfg=model_cfg)
    base_model.summary()
    phy_bind_inst = ToyPhyBindModel(base=base_model, random_start=8, sample_rate=10, sample_pts=32)
    compile_model(model=phy_bind_inst, net_compile_key=net_compile_key)
    # prepare data
    training_data_fit = np.expand_dims(training_data, axis=-1)
    fake_label_data = np.zeros(shape=(training_data.shape[0], 2, 1), dtype=np.float32)
    # train model and slope correction
    phy_bind_inst.fit(x=training_data_fit, y=fake_label_data, batch_size=16, epochs=200, verbose=1)
    training_data_res = np.reshape(training_data, newshape=(-1, training_data.shape[-1], 1))
    infer_result_arr = np.zeros(shape=(training_data_res.shape[0], 10, 1), dtype=np.float32)
    for i in range(10):
        training_data_slice = training_data_res[:, i::10, :]
        infer_result = base_model(training_data_slice)
        infer_result_arr[:, i, :] = infer_result
    x = np.linspace(0, 1, 10, endpoint=False)
    if verbose >= 1:
        plt.figure()
        for i in range(infer_result_arr.shape[0]):
            plt.plot(x, infer_result_arr[i, :, 0])
        plt.show()
    slope_arr = np.zeros(shape=infer_result_arr.shape[0], dtype=np.float32)
    for i in range(infer_result_arr.shape[0]):
        res = linregress(x, infer_result_arr[i, :, 0])
        slope_arr[i] = res.slope
    mean_slope = np.mean(slope_arr)
    # test model on test set
    test_data_res = np.reshape(test_data, newshape=(-1, test_data.shape[-1], 1))
    test_result = base_model(test_data_res)
    test_diff = test_result[0::2, 0] - test_result[1::2, 0]
    test_diff = test_diff * (1. / mean_slope) * timestep
    mean = np.mean(test_diff)
    std = np.std(test_diff)
    if verbose >= 1:
        plt.figure()
        plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(test_diff), float(mean), float(std)))
        _, bins, _ = plt.hist(test_diff.numpy(), density=True)
        x = np.linspace(bins[0], bins[-1], 101, endpoint=True)
        plt.plot(x, norm.pdf(x, loc=mean, scale=std), "r--")
        plt.show()
    return std


def apply_nn_quan(training_data, test_data, net_cfg_key="default_in32", net_compile_key="adam", timestep=8., verbose=0):
    # prepare model
    model_cfg = AVAILABLE_CONFIGS[net_cfg_key]
    base_model, q_cfg_dict = build_seq_model(cfg=model_cfg)
    base_model.summary()
    # prepare data
    training_data_fit = np.expand_dims(training_data, axis=-1)
    fake_label_data = np.zeros(shape=(training_data.shape[0], 2, 1), dtype=np.float32)
    # train base model
    phy_bind_inst = ToyPhyBindModel(base=base_model, random_start=8, sample_rate=10, sample_pts=32)
    compile_model(model=phy_bind_inst, net_compile_key=net_compile_key)
    phy_bind_inst.fit(x=training_data_fit, y=fake_label_data, batch_size=16, epochs=200, verbose=1)
    # quantization-aware training
    # q_aware stands for for quantization aware.
    with quantize_scope(q_cfg_dict):
        q_aware_model = quantize_apply(base_model)
    q_aware_model.summary()
    phy_bind_inst = ToyPhyBindModel(base=q_aware_model, random_start=8, sample_rate=10, sample_pts=32)
    compile_model(model=phy_bind_inst, net_compile_key=net_compile_key)
    phy_bind_inst.fit(x=training_data_fit, y=fake_label_data, batch_size=16, epochs=200, verbose=1)
    # slope correction
    training_data_res = np.reshape(training_data, newshape=(-1, training_data.shape[-1], 1))
    infer_result_arr = np.zeros(shape=(training_data_res.shape[0], 10, 1), dtype=np.float32)
    for i in range(10):
        training_data_slice = training_data_res[:, i::10, :]
        infer_result = q_aware_model(training_data_slice)
        infer_result_arr[:, i, :] = infer_result
    x = np.linspace(0, 1, 10, endpoint=False)
    if verbose >= 1:
        plt.figure()
        for i in range(infer_result_arr.shape[0]):
            plt.plot(x, infer_result_arr[i, :, 0])
        plt.show()
    slope_arr = np.zeros(shape=infer_result_arr.shape[0], dtype=np.float32)
    for i in range(infer_result_arr.shape[0]):
        res = linregress(x, infer_result_arr[i, :, 0])
        slope_arr[i] = res.slope
    mean_slope = np.mean(slope_arr)
    # test model on test set
    test_data_res = np.reshape(test_data, newshape=(-1, test_data.shape[-1], 1))
    test_result = q_aware_model(test_data_res)
    test_diff = test_result[0::2, 0] - test_result[1::2, 0]
    test_diff = test_diff * (1. / mean_slope) * timestep
    mean = np.mean(test_diff)
    std = np.std(test_diff)
    if verbose >= 1:
        plt.figure()
        plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(test_diff), float(mean), float(std)))
        _, bins, _ = plt.hist(test_diff.numpy(), density=True)
        x = np.linspace(bins[0], bins[-1], 101, endpoint=True)
        plt.plot(x, norm.pdf(x, loc=mean, scale=std), "r--")
        plt.show()
    return std


def work_routine(dirname, net_cfg_key="default_in32", use_quan=False, interp=10, folds=10, verbose=0):
    # generate dataset by interpolation
    adc_data_col = extract_data_multi(dirname=dirname, verbose=verbose)
    np.random.shuffle(adc_data_col)
    sample_len = adc_data_col.shape[-1]
    adc_data_res = np.reshape(adc_data_col, newshape=(-1, sample_len))
    adc_data_int = np.zeros(shape=(adc_data_res.shape[0], sample_len*interp), dtype=np.float32)
    x = np.arange(-1, sample_len, 1)
    for i in range(adc_data_res.shape[0]):
        y = np.concatenate((adc_data_res[i, 0:1], adc_data_res[i, :]), axis=0)
        func = interp1d(x, y)
        xnew = np.linspace(-1, sample_len-1, sample_len*interp, endpoint=False)
        ynew = func(xnew)
        if verbose >= 2:
            plt.figure()
            plt.plot(x, y, 'o', xnew, ynew, '-')
            plt.show()
        adc_data_int[i, :] = ynew
    adc_data_int = np.reshape(adc_data_int, newshape=adc_data_col.shape[:-1] + (sample_len*interp,))
    # run cross-validation
    sample_cnt = adc_data_col.shape[0]
    sample_cnt_div = sample_cnt // folds
    result_col = []
    for i in range(folds):
        test_index = np.arange(sample_cnt_div*i, sample_cnt_div*(i+1), 1, dtype=np.int32)
        test_vld = np.zeros(shape=sample_cnt, dtype=bool)
        test_vld[test_index] = True
        test_data = adc_data_col[test_vld]
        training_data = adc_data_int[np.logical_not(test_vld)]
        if use_quan:
            result = apply_nn_quan(training_data, test_data, net_cfg_key=net_cfg_key, verbose=verbose)
        else:
            result = apply_nn(training_data, test_data, net_cfg_key=net_cfg_key, verbose=verbose)
        result_col.append(result)
        print("fold %d: timing resolution: %.3f (%.3f)" % (i, float(result), float(result) / np.sqrt(2.)))
    # post-training analysis
    result_col = np.array(result_col)
    mean_result = np.mean(result_col)
    print("----------------------------")
    print("average: timing resolution: %.3f (%.3f)" % (float(mean_result), float(mean_result) / np.sqrt(2.)))


if __name__ == "__main__":
    work_routine(
        dirname="D:\\FPGAPrj\\nn_daq_trigger\\saved_data\\20220704",
        net_cfg_key="default_act16_in32",
        use_quan=True,
        verbose=0
    )
