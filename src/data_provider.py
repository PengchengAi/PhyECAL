import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

from src.util import update


def normalize_waveform(inputs, targets, norm_max_val=0.9, norm_min_val=-0.9):
    wave_max_val = np.max([np.max(inputs), np.max(targets)])
    wave_min_val = np.min([np.min(inputs), np.min(targets)])
    dynamic_range = wave_max_val - wave_min_val
    if dynamic_range < np.finfo(np.float32).eps:
        raise Exception("Normalization can not be performed.")
    norm_range = norm_max_val - norm_min_val
    inputs_normed = (inputs - wave_min_val) / dynamic_range * norm_range + norm_min_val
    targets_normed = (targets - wave_min_val) / dynamic_range * norm_range + norm_min_val
    return inputs_normed, targets_normed, wave_max_val, wave_min_val


def normalize_labels(labels, norm_max_val=0.9, norm_min_val=-0.9):
    axis = tuple(range(0, len(labels.shape) - 1))
    labels_max_val = np.max(labels, axis=axis)
    labels_min_val = np.min(labels, axis=axis)
    assert len(labels_max_val) == labels.shape[-1] and len(labels_min_val) == labels.shape[-1], "Dimension error."
    dynamic_range = labels_max_val - labels_min_val
    if dynamic_range < np.finfo(np.float32).eps:
        raise Exception("Normalization can not be performed.")
    norm_range = norm_max_val - norm_min_val
    # use broadcast to infer the shape of output
    labels_normed = (labels - labels_min_val) / dynamic_range * norm_range + norm_min_val
    return labels_normed, labels_max_val, labels_min_val


def process_npz_file(npz_file, norm_wave, norm_label, norm_max_val=0.9, norm_min_val=-0.9):
    if isinstance(npz_file, str) and os.path.exists(npz_file):
        content = np.load(npz_file)
    elif isinstance(npz_file, dict):
        content = npz_file
    else:
        raise Exception("The input argument npz_file is not valid.")
    count = content["count"]
    inputs = np.array(content["inputs"]).astype(np.float32)
    targets = np.array(content["targets"]).astype(np.float32)
    labels = np.array(content["labels"]).astype(np.float32)

    norm_dict = {}
    if norm_wave:
        inputs, targets, wave_max_val, wave_min_val = normalize_waveform(
            inputs, targets, norm_max_val=norm_max_val, norm_min_val=norm_min_val)
        norm_dict["wave_max_val"] = wave_max_val
        norm_dict["wave_min_val"] = wave_min_val
        norm_dict["norm_max_val"] = norm_max_val
        norm_dict["norm_min_val"] = norm_min_val
    else:
        wave_max_val = np.max([np.max(inputs), np.max(targets)])
        wave_min_val = np.min([np.min(inputs), np.min(targets)])
        norm_dict["wave_max_val"] = wave_max_val
        norm_dict["wave_min_val"] = wave_min_val

    if norm_label:
        labels, labels_max_val, labels_min_val = normalize_labels(
            labels, norm_max_val=norm_max_val, norm_min_val=norm_min_val)
        norm_dict["labels_max_val"] = labels_max_val
        norm_dict["labels_min_val"] = labels_min_val
        norm_dict["norm_max_val"] = norm_max_val
        norm_dict["norm_min_val"] = norm_min_val

    return count, inputs, targets, labels, norm_dict


class DataHandler(object):
    def __init__(self, npz_file, ratio=(4, 1), use_validation=False, norm_wave=True, norm_label=True, checkpoint=None,
                 reverse_wave=False, slicing_wave=None, origin_wave=False, ds_noise=None,
                 norm_max_val=0.9, norm_min_val=-0.9):
        assert isinstance(ratio, (list, tuple)), "Ratio is not valid."
        if use_validation:
            assert len(ratio) == 3, "Ratio length is not appropriate."
        else:
            assert len(ratio) == 2, "Ratio length is not appropriate."
        self._count, self._inputs, self._targets, self._labels, self._norm_dict = process_npz_file(
            npz_file, norm_wave, norm_label, norm_max_val=norm_max_val, norm_min_val=norm_min_val
        )
        self._ratio = ratio
        self._use_validation = use_validation
        self._norm_wave = norm_wave
        self._norm_label = norm_label
        ind_array = np.arange(0, self._count, 1, dtype=int)
        np.random.shuffle(ind_array)
        self._ind_array = ind_array
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                np.savez(checkpoint, ratio=self._ratio, use_validation=self._use_validation, ind_array=ind_array)
            else:
                npz_file_content = np.load(checkpoint)
                self._ratio = npz_file_content["ratio"]
                self._use_validation = bool(npz_file_content["use_validation"])
                self._ind_array = npz_file_content["ind_array"]
        self._reverse_wave = reverse_wave
        self._slicing_wave = slicing_wave
        self._origin_wave = origin_wave
        self._ds_noise = ds_noise
        self._masks = self.generate_dataset_masks()

    def get_norm_transform(self, item):
        if len(self._norm_dict) == 0:
            return None

        item_max_val = None
        item_min_val = None
        if item == "decoder" and "wave_max_val" in self._norm_dict:
            item_max_val = self._norm_dict["wave_max_val"]
            item_min_val = self._norm_dict["wave_min_val"]
        elif item == "time" and "labels_max_val" in self._norm_dict:
            item_max_val = self._norm_dict["labels_max_val"][0]
            item_min_val = self._norm_dict["labels_min_val"][0]
        elif item == "energy" and "labels_max_val" in self._norm_dict:
            item_max_val = self._norm_dict["labels_max_val"][1]
            item_min_val = self._norm_dict["labels_min_val"][1]

        if item_max_val is not None:
            norm_max_val = self._norm_dict["norm_max_val"]
            norm_min_val = self._norm_dict["norm_min_val"]
            dynamic_range = item_max_val - item_min_val
            norm_range = norm_max_val - norm_min_val
            scale = norm_range / dynamic_range
            shift = norm_min_val - item_min_val * scale
            return scale, shift

        return None

    def generate_dataset_masks(self):
        def mask_func(start, end, num, shuffled_ind):
            mask = np.zeros(shape=(num,), dtype=bool)
            selected_ind = shuffled_ind[start:end]
            mask[selected_ind] = True
            return mask

        count = self._count
        ind_array = self._ind_array

        sum_ratio = np.sum(self._ratio, dtype=int)
        if self._use_validation:
            end_validation = count * self._ratio[1] // sum_ratio
            end_test = count * (self._ratio[1] + self._ratio[2]) // sum_ratio
            end_train = count
            validation_mask = mask_func(0, end_validation, count, ind_array)
            test_mask = mask_func(end_validation, end_test, count, ind_array)
            train_mask = mask_func(end_test, end_train, count, ind_array)
            return train_mask, validation_mask, test_mask
        else:
            end_test = count * self._ratio[1] // sum_ratio
            end_train = count
            test_mask = mask_func(0, end_test, count, ind_array)
            train_mask = mask_func(end_test, end_train, count, ind_array)
            return train_mask, test_mask

    def adapt_dataset_noise(self, ds_noise):
        if self._norm_wave:
            dynamic_range = self._norm_dict["norm_max_val"] - self._norm_dict["norm_min_val"]
        else:
            dynamic_range = self._norm_dict["wave_max_val"] - self._norm_dict["wave_min_val"]
        return ds_noise * dynamic_range

    def generate_training_dataset(self, slice_label=None):
        train_mask = self._masks[0]

        count = np.count_nonzero(train_mask)
        inputs = self._inputs[train_mask, ...]
        targets = self._targets[train_mask, ...]
        labels = self._labels[train_mask, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            labels = labels[..., slice_label]

        inputs = np.expand_dims(inputs, axis=-1)
        targets = np.expand_dims(targets, axis=-1)

        return {
            "count": count,
            "inputs": inputs,
            "targets": targets,
            "labels": labels
        }

    def generate_validation_dataset(self, slice_label=None):
        if not self._use_validation:
            raise Exception("No validation dataset is specified.")
        validation_mask = self._masks[1]

        count = np.count_nonzero(validation_mask)
        inputs = self._inputs[validation_mask, ...]
        targets = self._targets[validation_mask, ...]
        labels = self._labels[validation_mask, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            labels = labels[..., slice_label]

        inputs = np.expand_dims(inputs, axis=-1)
        targets = np.expand_dims(targets, axis=-1)

        return {
            "count": count,
            "inputs": inputs,
            "targets": targets,
            "labels": labels
        }

    def generate_test_dataset(self, slice_label=None):
        if self._use_validation:
            test_mask = self._masks[2]
        else:
            test_mask = self._masks[1]

        count = np.count_nonzero(test_mask)
        inputs = self._inputs[test_mask, ...]
        targets = self._targets[test_mask, ...]
        labels = self._labels[test_mask, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            labels = labels[..., slice_label]

        inputs = np.expand_dims(inputs, axis=-1)
        targets = np.expand_dims(targets, axis=-1)

        return {
            "count": count,
            "inputs": inputs,
            "targets": targets,
            "labels": labels
        }

    def generate_test_val_dataset(self, slice_label=None):
        if self._use_validation:
            test_mask = self._masks[2]
            validation_mask = self._masks[1]
            test_val_mask = np.logical_or(test_mask, validation_mask)
        else:
            test_val_mask = self._masks[1]

        count = np.count_nonzero(test_val_mask)
        inputs = self._inputs[test_val_mask, ...]
        targets = self._targets[test_val_mask, ...]
        labels = self._labels[test_val_mask, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            labels = labels[..., slice_label]

        inputs = np.expand_dims(inputs, axis=-1)
        targets = np.expand_dims(targets, axis=-1)

        return {
            "count": count,
            "inputs": inputs,
            "targets": targets,
            "labels": labels
        }


class CVDataHandler(DataHandler):
    def __init__(self, folds, npz_file, **kwargs):
        super(CVDataHandler, self).__init__(npz_file, **kwargs)
        self._folds = folds
        self._cnt_div = self._count // self._folds

    def generate_fold_dataset(self, fold_index, slice_label=None):
        assert 0 <= fold_index <= self._folds - 1, "Fold index out of range."
        test_index = np.arange(self._cnt_div * fold_index, self._cnt_div * (fold_index + 1), 1, dtype=np.int32)
        test_vld = np.zeros(shape=self._count, dtype=bool)
        test_vld[test_index] = True

        test_count = np.count_nonzero(test_vld)
        test_inputs = self._inputs[test_vld, ...]
        test_targets = self._targets[test_vld, ...]
        test_labels = self._labels[test_vld, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            test_labels = test_labels[..., slice_label]

        test_inputs = np.expand_dims(test_inputs, axis=-1)
        test_targets = np.expand_dims(test_targets, axis=-1)

        test_dataset = {
            "count": test_count,
            "inputs": test_inputs,
            "targets": test_targets,
            "labels": test_labels
        }

        train_vld = np.logical_not(test_vld)

        train_count = np.count_nonzero(train_vld)
        train_inputs = self._inputs[train_vld, ...]
        train_targets = self._targets[train_vld, ...]
        train_labels = self._labels[train_vld, ...]

        if slice_label is not None:
            if not isinstance(slice_label, (tuple, list)):
                slice_label = (slice_label,)
            train_labels = train_labels[..., slice_label]

        train_inputs = np.expand_dims(train_inputs, axis=-1)
        train_targets = np.expand_dims(train_targets, axis=-1)

        train_dataset = {
            "count": train_count,
            "inputs": train_inputs,
            "targets": train_targets,
            "labels": train_labels
        }

        return train_dataset, test_dataset


def prepare_data_inst_npz(config_file, upd_dict=None, data_key="bind", amount=None, debug=False, store_gen_data=True):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    if upd_dict is not None:
        cfg = update(cfg, upd_dict)
        print("Configuration has been updated with the dictionary:", upd_dict)

    if store_gen_data:
        if not os.path.exists(cfg["supp"]["save_dir"]):
            os.makedirs(cfg["supp"]["save_dir"])
        gen_path = os.path.join(
            cfg["supp"]["save_dir"],
            "%s-%s_data.npz" % (cfg["supp"]["save_prefix"], data_key)
        )
    else:
        gen_path = None

    if gen_path is not None and os.path.exists(gen_path):
        cfg["dataset"][data_key]["npz_file"] = gen_path
        dh_inst = DataHandler(**(cfg["dataset"][data_key]))
        return dh_inst

    npz_file_list = cfg["data_raw_file"][data_key]

    npz_content_list = []
    total_cnt = 0
    for npz_file in npz_file_list:
        content = np.load(npz_file)
        total_cnt += int(content["count"])
        npz_content_list.append(content)

    print("Total %d examples." % total_cnt)
    if amount is None:
        amount = total_cnt

    sig_cfg = cfg["signal_ch"]

    if "channel_count" in sig_cfg:
        channel_count = sig_cfg["channel_count"]
    else:
        channel_count = int((npz_content_list[0]["data"]).shape[1])

    sample_every = sig_cfg["sample_every"]
    sample_points = sig_cfg["sample_points"]
    sample_length = sample_every * sample_points

    if not isinstance(sig_cfg["sample_start"], (list, tuple)):
        sample_start = [sig_cfg["sample_start"] for _ in range(len(npz_content_list))]
    else:
        sample_start = sig_cfg["sample_start"]
        assert len(sample_start) == len(npz_content_list), "The Length of sample start is not expected."

    file_index_multiple = sig_cfg["file_index_multiple"]

    if "random_origin" in sig_cfg:
        random_origin = sig_cfg["random_origin"]
    else:
        random_origin = [0, 0]

    if "shift_start" in sig_cfg:
        shift_start = sig_cfg["shift_start"]
    else:
        shift_start = [-1, 1]

    if "baseline_range" in sig_cfg:
        if not isinstance(sig_cfg["baseline_range"][0], (tuple, list)):
            baseline_range = [sig_cfg["baseline_range"] for _ in range(len(npz_content_list))]
        else:
            baseline_range = sig_cfg["baseline_range"]
            assert len(baseline_range) == len(npz_content_list), "The Length of baseline range is not expected."
    else:
        baseline_range = [[0, sample_start[i]] for i in range(len(npz_content_list))]

    inputs_data = np.zeros(shape=(amount, channel_count, sample_points), dtype=np.float64)
    targets_data = np.zeros(shape=(amount, channel_count, sample_points), dtype=np.float64)
    labels_data = np.zeros(shape=(amount, channel_count, 1), dtype=np.float64)
    index_array = np.zeros(shape=amount, dtype=int)

    cont_index = 0
    data_index = 0
    print("Start to process No. %d npz file." % cont_index)
    for i in range(amount):
        i_wave = npz_content_list[cont_index]["data"][data_index, ...]
        i_index = npz_content_list[cont_index]["index"][data_index]
        # cancel baseline
        for j in range(channel_count):
            j_baseline = np.mean(i_wave[j, baseline_range[cont_index][0]:baseline_range[cont_index][1]], axis=None)
            i_wave[j, :] = i_wave[j, :] - j_baseline
        # start of sub-sampling
        if random_origin[0] >= random_origin[1]:
            i_origin = random_origin[0]
        else:
            i_origin = np.random.randint(low=random_origin[0], high=random_origin[1], size=None, dtype=int)
        i_start = i_origin + sample_start[cont_index]
        # random shift
        i_shift = np.random.choice(shift_start, size=channel_count, replace=True)
        for j in range(channel_count):
            j_loc = i_start + i_shift[j]
            inputs_data[i, j, :] = i_wave[j, j_loc:j_loc + sample_length:sample_every]
            targets_data[i, j, :] = i_wave[j, i_start:i_start + sample_length:sample_every]
            labels_data[i, j, :] = -i_shift[j]
        # record index
        index_array[i] = i_index + file_index_multiple * cont_index
        # debug
        if debug:
            print("label:", labels_data[i, :, 0].tolist())
            plt.figure()
            for j in range(channel_count):
                plt.plot(inputs_data[i, j, :], label="input ch%d" % j)
            plt.legend()
            plt.show()
        # increment indexes
        data_index += 1
        if data_index == int(npz_content_list[cont_index]["count"]):
            data_index = 0
            cont_index += 1
            if cont_index == len(npz_content_list):
                cont_index = 0
                print("Rollback to No. %d npz file." % cont_index)
            else:
                print("Continue to process No. %d npz file." % cont_index)

    data_dict = {
        "count": amount,
        "index": index_array,
        "inputs": inputs_data,
        "targets": targets_data,
        "labels": labels_data
    }

    if gen_path is not None and not os.path.exists(gen_path):
        np.savez(
            gen_path,
            **data_dict
        )

    cfg["dataset"][data_key]["npz_file"] = data_dict
    dh_inst = DataHandler(**(cfg["dataset"][data_key]))
    return dh_inst


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


def prepare_data_inst_cv(config_file, upd_dict=None, data_key="toy", file_cnt=10, verbose=0, store_gen_data=True):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    if upd_dict is not None:
        cfg = update(cfg, upd_dict)
        print("Configuration has been updated with the dictionary:", upd_dict)

    if store_gen_data:
        if not os.path.exists(cfg["supp"]["save_dir"]):
            os.makedirs(cfg["supp"]["save_dir"])
        gen_path = os.path.join(
            cfg["supp"]["save_dir"],
            "%s-%s_data.npz" % (cfg["supp"]["save_prefix"], data_key)
        )
    else:
        gen_path = None

    if gen_path is not None and os.path.exists(gen_path):
        cfg["dataset"][data_key]["npz_file"] = gen_path
        dh_inst = CVDataHandler(**(cfg["dataset"][data_key]))
        return dh_inst

    sig_cfg = cfg["signal_ila"]

    # generate dataset by interpolation
    adc_data_col = ila_extract_data_multi(dirname=sig_cfg["dirname"], file_cnt=file_cnt, verbose=verbose)
    np.random.shuffle(adc_data_col)
    sample_len = adc_data_col.shape[-1]
    adc_data_res = np.reshape(adc_data_col, newshape=(-1, sample_len))
    adc_data_int = np.zeros(shape=(adc_data_res.shape[0], sample_len * sig_cfg["interp"]), dtype=np.float32)
    x = np.arange(-1, sample_len, 1)
    for i in range(adc_data_res.shape[0]):
        y = np.concatenate((adc_data_res[i, 0:1], adc_data_res[i, :]), axis=0)
        func = interp1d(x, y)
        xnew = np.linspace(-1, sample_len - 1, sample_len * sig_cfg["interp"], endpoint=False)
        ynew = func(xnew)
        if verbose >= 2:
            plt.figure()
            plt.plot(x, y, 'o', xnew, ynew, '-')
            plt.show()
        adc_data_int[i, :] = ynew
    adc_data_int = np.reshape(adc_data_int, newshape=adc_data_col.shape[:-1] + (sample_len * sig_cfg["interp"],))
    sample_cnt = adc_data_col.shape[0]

    data_dict = {
        "count": sample_cnt,
        "inputs": adc_data_int,
        "targets": adc_data_col,
        "labels": np.zeros(shape=(sample_cnt, 2, 1), dtype=np.float32)
    }

    if gen_path is not None and not os.path.exists(gen_path):
        np.savez(
            gen_path,
            **data_dict
        )

    cfg["dataset"][data_key]["npz_file"] = data_dict
    dh_inst = CVDataHandler(**(cfg["dataset"][data_key]))
    return dh_inst
