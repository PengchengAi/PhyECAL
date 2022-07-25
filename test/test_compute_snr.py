import os

import numpy as np
import pandas as pd


def ila_extract_data(path, verbose=0):
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


def ila_extract_data_multi(dirname, file_cnt=10, file_prefix="iladata", verbose=0):
    adc_data_col = []
    for i in range(file_cnt):
        filename = file_prefix + str(i+1) + ".csv"
        file_path = os.path.join(dirname, filename)
        adc_data = ila_extract_data(path=file_path, verbose=verbose)
        adc_data_col.append(adc_data)
    adc_data_col = np.concatenate(adc_data_col, axis=0)
    return adc_data_col


def main(dirname, file_cnt=10):
    adc_data_col = ila_extract_data_multi(dirname=dirname, file_cnt=file_cnt)
    # noise
    baseline = adc_data_col[:, :, 0].astype(np.float32)
    baseline_std = np.std(baseline, axis=0)
    # signal
    baseline_reshape = np.expand_dims(baseline, axis=-1)
    adj_waveform = (adc_data_col - baseline_reshape).astype(np.float32)
    amplitude = np.max(adj_waveform, axis=-1)
    avg_ampitude = np.mean(amplitude, axis=0)
    # SNR
    snr = avg_ampitude / baseline_std
    snr_db = 20 * np.log10(snr)
    # print results
    print("Signal-Noise Ratio results:")
    print("-------------------------------------")
    print("baseline noise std.:", baseline_std.tolist())
    print("average amplitude:", avg_ampitude.tolist())
    print("SNR:", snr.tolist())
    print("SNR in dB:", snr_db.tolist())


if __name__ == "__main__":
    main(
        dirname="D:\\FPGAPrj\\nn_daq_trigger\\saved_data\\20220724",
        file_cnt=100
    )
