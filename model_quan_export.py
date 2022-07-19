import os
import yaml
import inspect

import h5py
import numpy as np
from tensorflow.keras.models import load_model, Model
from scipy.stats import linregress

from conf.net_config import AVAILABLE_CONFIGS
from src.data_provider import prepare_data_inst_cv, prepare_data_inst_npz


def get_int_params(weights_dict, val_key, min_key, max_key, bits, narrow_range=False, scale=1, verbose=0):
    if val_key is not None:
        if isinstance(val_key, str):
            val = weights_dict[val_key]
        else:
            val = val_key
    else:
        val = None
    if isinstance(min_key, str):
        min_t = weights_dict[min_key]
    else:
        min_t = min_key
    if isinstance(max_key, str):
        max_t = weights_dict[max_key]
    else:
        max_t = max_key
    if scale is not None:
        min_t = min_t * scale
        max_t = max_t * scale
    if narrow_range:
        intervals = 2 ** bits - 2
    else:
        intervals = 2 ** bits - 1
    scale_quan = (max_t - min_t) / intervals
    min_adj = scale_quan * np.round(min_t / scale_quan)
    max_adj = max_t + min_adj - min_t
    if val is not None:
        val_clamp = np.clip(val, a_min=min_adj, a_max=max_adj)
        val_quan = np.round((val_clamp - min_adj) / scale_quan).astype(dtype=np.int64)
    else:
        val_quan = None
    zero_quan = np.round((0. - min_adj) / scale_quan).astype(dtype=np.int64)
    if verbose >= 2 and val_quan is not None:
        print("value:", val_quan)
    if verbose >= 1:
        print("scale:", scale_quan)
        print("original min:", min_t, "adjusted min:", min_adj)
        print("original max:", max_t, "adjusted max:", max_adj)
        print("zero:", zero_quan)
    return val_quan, zero_quan, scale_quan, min_adj, max_adj


def approx_m_with_bits(m_value, bits=16):
    m_temp_val = m_value.copy()
    n = 0
    while m_temp_val < 0.5:
        m_temp_val = m_temp_val * 2
        n = n + 1
    m_temp_val = m_temp_val * (2 ** (bits - 1))
    m_temp_val = np.round(m_temp_val).astype(np.int64)
    return m_temp_val, n + bits - 1


def rounding_discard_bits(value, bits, round_to_even=True):
    negative = False
    if value < 0:
        value = -value
        negative = True
    value_front = value // (2 ** bits) % 2
    value_on = value // (2 ** (bits - 1)) % 2
    value_suf = value % (2 ** (bits - 1))
    if round_to_even:
        if np.equal(value_suf, 0) and np.equal(value_on, 1):
            if np.equal(value_front, 0):
                result = value // (2 ** bits)
            else:
                result = value // (2 ** bits) + 1
        else:
            if np.equal(value_on, 0):
                result = value // (2 ** bits)
            else:
                result = value // (2 ** bits) + 1
    else:
        if np.equal(value_on, 0):
            result = value // (2 ** bits)
        else:
            result = value // (2 ** bits) + 1
    if negative:
        result = -result
    return result


def conv1d_sim(val_input, val_conv_kernel, zero_input=None, zero_conv_kernel=None):
    output_length = val_input.shape[1] // 2
    conv_result = np.zeros(shape=(val_input.shape[0], output_length, val_conv_kernel.shape[2]), dtype=np.int64)
    for b in range(val_input.shape[0]):
        for oc in range(val_conv_kernel.shape[2]):
            temp_result = np.zeros(shape=(output_length,), dtype=np.int64)
            for ic in range(val_input.shape[2]):
                if zero_input is not None:
                    input_line_deduced = val_input[b, :, ic] - zero_input
                else:
                    input_line_deduced = val_input[b, :, ic]
                if zero_conv_kernel is not None:
                    val_conv_kernel_deduced = val_conv_kernel[:, ic, oc] - zero_conv_kernel
                else:
                    val_conv_kernel_deduced = val_conv_kernel[:, ic, oc]
                input_line_ext = np.concatenate(
                    (np.array(0, dtype=np.int64), input_line_deduced, np.array(0, dtype=np.int64)),
                    axis=None
                )
                for ol in range(output_length):
                    temp_result[ol] = temp_result[ol] + np.sum(
                        input_line_ext[ol * 2:ol * 2 + 4] * val_conv_kernel_deduced)
            conv_result[b, :, oc] = temp_result

    return conv_result


def fc1d_sim(val_input, val_fc_kernel, zero_input=None, zero_fc_kernel=None):
    if zero_input is not None:
        val_input_deduced = val_input - zero_input
    else:
        val_input_deduced = val_input
    if zero_fc_kernel is not None:
        val_fc_kernel_deduced = val_fc_kernel - zero_fc_kernel
    else:
        val_fc_kernel_deduced = val_fc_kernel
    fc_result = np.matmul(val_input_deduced, val_fc_kernel_deduced)

    return fc_result


def model_export(config_file, param_export_path, hdf5_path, force_override=False, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    if not force_override and os.path.exists(hdf5_path):
        print("The HDF5 file has already existed.")
        return False
    else:
        dirname = os.path.dirname(os.path.abspath(hdf5_path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    exp_cfg = cfg["export"]
    if "upd_kwargs" in exp_cfg:
        kwargs.update(exp_cfg["upd_kwargs"])
    kwargs["config_file"] = config_file

    if "use_toy" in kwargs and kwargs["use_toy"]:
        prepare_data_func = prepare_data_inst_cv
    else:
        prepare_data_func = prepare_data_inst_npz

    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_func)[0]}
    data_handler = prepare_data_func(**data_kwargs)

    weights_dict = np.load(param_export_path)

    # start processing datasets
    f_root = h5py.File(hdf5_path, mode="w")
    f_data = f_root.create_group("dataset")

    zero_input = None
    scale_input = None
    max_input = None
    min_input = None
    for ds in ["val", "test", "train", "testval"]:
        if ds == "val":
            data_dict = data_handler.generate_validation_dataset()
        elif ds == "test":
            data_dict = data_handler.generate_test_dataset()
        elif ds == "train":
            data_dict = data_handler.generate_training_dataset()
        elif ds == "testval":
            data_dict = data_handler.generate_test_val_dataset()
        else:
            raise Exception("Wrong key.")

        f_ds = f_data.create_group(ds)

        for key in ["count", "inputs", "targets", "labels"]:
            if key == "count":
                ds_ds = f_ds.create_dataset(key, shape=(1,), dtype=np.int64)
                ds_ds[0] = data_dict[key]
            elif key == "inputs" or key == "targets":
                ds_ds = f_ds.create_dataset(key, shape=data_dict[key].shape, dtype=np.int64)

                val_input, zero_input, scale_input, min_input, max_input = get_int_params(
                    weights_dict=weights_dict,
                    val_key=data_dict[key],
                    min_key='quantize_layer/quantize_layer_min:0',
                    max_key='quantize_layer/quantize_layer_max:0',
                    bits=exp_cfg["input_bits"],
                    narrow_range=False,
                    scale=exp_cfg["input_scale"],
                    verbose=1
                )
                ds_ds[:] = val_input - zero_input
                ds_ds.attrs["zero"] = zero_input
                ds_ds.attrs["scale"] = scale_input
                ds_ds.attrs["min"] = min_input
                ds_ds.attrs["max"] = max_input
            elif key == "labels":
                ds_ds = f_ds.create_dataset(key, shape=data_dict[key].shape, dtype=np.float32)
                ds_ds[:] = data_dict[key]
            else:
                raise Exception("Wrong key.")
    f_data.attrs["zero"] = zero_input
    f_data.attrs["scale"] = scale_input
    f_data.attrs["min"] = min_input
    f_data.attrs["max"] = max_input

    # store normalization parameters
    f_norm = f_data.create_group("normalization")
    for item in ["decoder", "time", "energy"]:
        ret = data_handler.get_norm_transform(item=item)
        if ret is not None:
            scale, shift = ret
            f_item = f_norm.create_group(item)
            ds_scale = f_item.create_dataset("scale", shape=(1,), dtype=np.float32)
            ds_shift = f_item.create_dataset("shift", shape=(1,), dtype=np.float32)
            ds_scale[0] = scale
            ds_shift[0] = shift

    # start processing network architecture
    net_cfg = AVAILABLE_CONFIGS[kwargs["net_cfg_key"]]
    conv_layer_num = len(net_cfg["encoder_layer_spec"])
    fc_layer_num = len(net_cfg["regression_spec"])
    output_spec_names = [elem[2] for elem in net_cfg["slicing_spec"]]

    f_net = f_root.create_group("network")
    dt = h5py.special_dtype(vlen=str)
    ds_output = f_net.create_dataset("output_spec", shape=len(output_spec_names), dtype=dt)
    for i in range(len(output_spec_names)):
        ds_output[i] = output_spec_names[i]
        print("string saved:", ds_output[i])
    for i in range(len(output_spec_names)):
        assert ds_output[:][i].decode("UTF-8") == output_spec_names[i], \
            "Something wrong with the string format (%s, %s)." % (ds_output[:][i].decode("UTF-8"), output_spec_names[i])

    scale_conv_act = None
    for i in range(conv_layer_num):
        f_conv = f_net.create_group("enc_conv_%d" % i)
        val_conv_kernel, zero_conv_kernel, scale_conv_kernel, min_conv_kernel, max_conv_kernel = get_int_params(
            weights_dict=weights_dict,
            val_key='enc_conv_%d/kernel:0' % i,
            min_key='quant_enc_conv_%d/kernel_min:0' % i,
            max_key='quant_enc_conv_%d/kernel_max:0' % i,
            bits=exp_cfg["conv_kernel_bits"],
            narrow_range=False,
            scale=exp_cfg["conv_kernel_scale"],
            verbose=1
        )
        ds_knl = f_conv.create_dataset("kernel", shape=val_conv_kernel.shape, dtype=np.int64)
        ds_knl[:] = val_conv_kernel - zero_conv_kernel
        ds_knl.attrs["zero"] = zero_conv_kernel
        ds_knl.attrs["scale"] = scale_conv_kernel
        ds_knl.attrs["min"] = min_conv_kernel
        ds_knl.attrs["max"] = max_conv_kernel

        val_conv_bias = weights_dict['enc_conv_%d/bias:0' % i]
        zero_conv_bias = np.array(0, dtype=np.int64)
        if i == 0:
            scale_last_layer = scale_input
        else:
            scale_last_layer = scale_conv_act
        scale_conv_bias = scale_last_layer * scale_conv_kernel
        val_conv_bias = np.round(val_conv_bias / scale_conv_bias + zero_conv_bias).astype(dtype=np.int64)
        ds_bias = f_conv.create_dataset("bias", shape=val_conv_bias.shape, dtype=np.int64)
        ds_bias[:] = val_conv_bias - zero_conv_bias
        ds_bias.attrs["zero"] = zero_conv_bias
        ds_bias.attrs["scale"] = scale_conv_bias

        _, zero_conv_act, scale_conv_act, min_conv_act, max_conv_act = get_int_params(
            weights_dict=weights_dict,
            val_key=None,
            min_key='quant_enc_conv_%d/post_activation_min:0' % i,
            max_key='quant_enc_conv_%d/post_activation_max:0' % i,
            bits=exp_cfg["conv_act_bits"],
            narrow_range=False,
            scale=exp_cfg["conv_act_scale"],
            verbose=1
        )
        m_conv_act = scale_conv_bias / scale_conv_act
        m_conv_act_quan, m_conv_act_shift = approx_m_with_bits(m_value=m_conv_act, bits=16)
        f_act = f_conv.create_group("activation")
        f_act.attrs["zero"] = zero_conv_act
        f_act.attrs["scale"] = scale_conv_act
        f_act.attrs["min"] = min_conv_act
        f_act.attrs["max"] = max_conv_act
        ds_rescale = f_act.create_dataset("rescale", shape=(1,), dtype=np.int64)
        ds_shift = f_act.create_dataset("shift", shape=(1,), dtype=np.int64)
        ds_rescale[0] = m_conv_act_quan
        ds_shift[0] = m_conv_act_shift

    scale_fc_act = None
    for i in range(fc_layer_num):
        f_fc = f_net.create_group("reg_fc_%d" % i)
        val_fc_kernel, zero_fc_kernel, scale_fc_kernel, min_fc_kernel, max_fc_kernel = get_int_params(
            weights_dict=weights_dict,
            val_key='reg_fc_%d/kernel:0' % i,
            min_key='quant_reg_fc_%d/kernel_min:0' % i,
            max_key='quant_reg_fc_%d/kernel_max:0' % i,
            bits=exp_cfg["fc_kernel_bits"],
            narrow_range=False,
            scale=exp_cfg["fc_kernel_scale"],
            verbose=1
        )
        ds_knl = f_fc.create_dataset("kernel", shape=val_fc_kernel.shape, dtype=np.int64)
        ds_knl[:] = val_fc_kernel - zero_fc_kernel
        ds_knl.attrs["zero"] = zero_fc_kernel
        ds_knl.attrs["scale"] = scale_fc_kernel
        ds_knl.attrs["min"] = min_fc_kernel
        ds_knl.attrs["max"] = max_fc_kernel

        val_fc_bias = weights_dict['reg_fc_%d/bias:0' % i]
        zero_fc_bias = np.array(0, dtype=np.int64)
        if i == 0:
            scale_last_layer = scale_conv_act
        else:
            scale_last_layer = scale_fc_act
        scale_fc_bias = scale_last_layer * scale_fc_kernel
        val_fc_bias = np.round(val_fc_bias / scale_fc_bias + zero_fc_bias).astype(dtype=np.int64)
        ds_bias = f_fc.create_dataset("bias", shape=val_fc_bias.shape, dtype=np.int64)
        ds_bias[:] = val_fc_bias - zero_fc_bias
        ds_bias.attrs["zero"] = zero_fc_bias
        ds_bias.attrs["scale"] = scale_fc_bias

        if i != fc_layer_num - 1:
            _, zero_fc_act, scale_fc_act, min_fc_act, max_fc_act = get_int_params(
                weights_dict=weights_dict,
                val_key=None,
                min_key='quant_reg_fc_%d/post_activation_min:0' % i,
                max_key='quant_reg_fc_%d/post_activation_max:0' % i,
                bits=exp_cfg["fc_act_bits"],
                narrow_range=False,
                scale=exp_cfg["fc_act_scale"],
                verbose=1
            )
            m_fc_act = scale_fc_bias / scale_fc_act
            m_fc_act_quan, m_fc_act_shift = approx_m_with_bits(m_value=m_fc_act, bits=16)
            f_act = f_fc.create_group("activation")
            f_act.attrs["zero"] = zero_fc_act
            f_act.attrs["scale"] = scale_fc_act
            f_act.attrs["min"] = min_fc_act
            f_act.attrs["max"] = max_fc_act
            ds_rescale = f_act.create_dataset("rescale", shape=(1,), dtype=np.int64)
            ds_shift = f_act.create_dataset("shift", shape=(1,), dtype=np.int64)
            ds_rescale[0] = m_fc_act_quan
            ds_shift[0] = m_fc_act_shift

    print("HDF5 file generation has finished.")
    f_root.close()
    return True


def size_both_metric(result, count):
    assert result.shape[0] == count and result.shape[1] == 2, "Shape mismatches."
    diff = result[:, 0] - result[:, 1]
    # compute metrics
    mean = np.mean(diff)
    std = np.std(diff)
    return mean, std


def size_many_metric(result, count):
    assert result.shape[0] == count and result.shape[1] >= 4, "Shape mismatches."
    group_size = result.shape[1]
    # linear regression
    x = np.arange(group_size, dtype=np.float32)
    total_residual_sq = 0.
    for i in range(count):
        i_res = linregress(x, result[i, :])
        intercept, slope = i_res.intercept, i_res.slope
        residual_sq = np.sum((result[i, :] - (intercept + x * slope)) ** 2)
        total_residual_sq += residual_sq
    # compute metrics
    mean = 0.
    std = np.sqrt(total_residual_sq / (count * group_size))
    return mean, std


def simulated_inference(config_file, hdf5_path, model_result_path, in_set="testval", round_to_even=True, debug=False,
                        assertion=False, exclude_outliers=False, export_fmap=None):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]

    if not os.path.exists(hdf5_path):
        raise Exception("The HDF5 file path is not valid.")

    f_root = h5py.File(hdf5_path, mode="r")
    f_data = f_root["dataset"]
    f_ds = f_data[in_set]

    inputs = f_ds["targets"][:]
    sample_cnt = inputs.shape[0]
    group_size = inputs.shape[1]

    inputs_reshape = np.reshape(inputs, newshape=(sample_cnt*group_size,)+inputs.shape[-2::1])

    # create valid mask for excluding outliers
    if exclude_outliers:
        valid_mask = np.ones(shape=inputs_reshape.shape[0], dtype=bool)
    else:
        valid_mask = None

    if export_fmap is not None:
        fmap_dict = {
            "inputs": inputs_reshape
        }
    else:
        fmap_dict = {}

    f_net = f_root["network"]
    output_spec_names = f_net["output_spec"][:]

    last_fmap = inputs_reshape
    last_layer_name = None
    # infer with convolution layers if there are any
    conv_layers = [key for key in f_net.keys() if str(key).startswith("enc_conv")]
    if len(conv_layers) > 0:
        for i in range(len(conv_layers)):
            f_conv = f_net["enc_conv_%d" % i]
            # perform convolution operation
            kernel = f_conv["kernel"][:]
            bias = f_conv["bias"][:]
            if assertion:
                assert np.all(kernel >= -128) and np.all(kernel <= 127), "Kernel assertion failed in conv layer %d" % i
                assert np.all(bias >= np.iinfo(np.int32).min) and np.all(bias <= np.iinfo(np.int32).max), \
                    "Bias assertion failed in conv layer %d" % i
            conv_res = conv1d_sim(val_input=last_fmap, val_conv_kernel=kernel)
            conv_bias_res = conv_res + bias[np.newaxis, np.newaxis, :]
            # post-process
            if "activation" in f_conv.keys():
                f_act = f_conv["activation"]
                rescale = f_act["rescale"][0]
                shift = f_act["shift"][0]
                conv_bias_relu_res = np.clip(conv_bias_res, a_min=0, a_max=None)
                rdb_conv_v = np.vectorize(lambda x: rounding_discard_bits(x, bits=shift, round_to_even=round_to_even))
                conv_act_res = rdb_conv_v(conv_bias_relu_res * rescale)
                if assertion and not exclude_outliers:
                    assert np.all(conv_act_res >= 0) and np.all(conv_act_res <= 255), \
                        "Activation assertion failed in conv layer %d (%d)" % (i, np.count_nonzero(conv_act_res > 255))
                if exclude_outliers:
                    range_valid = np.all(conv_act_res <= 255, axis=tuple(np.arange(1, len(conv_act_res.shape))))
                    exclude_cnt = np.count_nonzero(np.logical_not(range_valid))
                    if exclude_cnt > 0:
                        print("Exclude %d examples in conv layer %d" % (exclude_cnt, i))
                        valid_mask = np.logical_and(valid_mask, range_valid)
                if debug:
                    print(f_act.attrs["zero"], f_act.attrs["scale"])
                    print("enc_conv_%d" % i, conv_act_res[:1].tolist())
            else:
                conv_act_res = conv_bias_res
                if assertion and not exclude_outliers:
                    assert np.all(conv_act_res >= np.iinfo(np.int32).min) and \
                        np.all(conv_act_res <= np.iinfo(np.int32).max), \
                        "Activation assertion failed in conv layer %d" % i
                if exclude_outliers:
                    range_valid = np.all(
                        np.logical_and(conv_act_res >= np.iinfo(np.int32).min, conv_act_res <= np.iinfo(np.int32).max),
                        axis=tuple(np.arange(1, len(conv_act_res.shape))))
                    exclude_cnt = np.count_nonzero(np.logical_not(range_valid))
                    if exclude_cnt > 0:
                        print("Exclude %d examples in conv layer %d" % (exclude_cnt, i))
                        valid_mask = np.logical_and(valid_mask, range_valid)
                if debug:
                    conv_act_res_recover = conv_act_res[:1] * f_conv["bias"].attrs["scale"]
                    print("enc_conv_%d" % i, conv_act_res_recover.tolist())
            if export_fmap is not None:
                fmap_dict["enc_conv_output_%d" % i] = conv_act_res
            last_fmap = conv_act_res
            last_layer_name = "enc_conv_%d" % i
    # flatten between convolution layers and fully-connected layers
    last_fmap = np.reshape(last_fmap, newshape=(last_fmap.shape[0], -1))
    # infer with fully-connected layers if there are any
    fc_layers = [key for key in f_net.keys() if str(key).startswith("reg_fc")]
    if len(fc_layers) > 0:
        for i in range(len(fc_layers)):
            f_fc = f_net["reg_fc_%d" % i]
            # perform fc operation
            kernel = f_fc["kernel"][:]
            bias = f_fc["bias"][:]
            if assertion:
                assert np.all(kernel >= -128) and np.all(kernel <= 127), "Kernel assertion failed in fc layer %d" % i
                assert np.all(bias >= np.iinfo(np.int32).min) and np.all(bias <= np.iinfo(np.int32).max), \
                    "Bias assertion failed in fc layer %d" % i
            fc_res = fc1d_sim(val_input=last_fmap, val_fc_kernel=kernel)
            fc_bias_res = fc_res + bias[np.newaxis, :]
            # post-process
            if "activation" in f_fc.keys():
                f_act = f_fc["activation"]
                rescale = f_act["rescale"][0]
                shift = f_act["shift"][0]
                fc_bias_relu_res = np.clip(fc_bias_res, a_min=0, a_max=None)
                rdb_conv_v = np.vectorize(lambda x: rounding_discard_bits(x, bits=shift, round_to_even=round_to_even))
                fc_act_res = rdb_conv_v(fc_bias_relu_res * rescale)
                if assertion and not exclude_outliers:
                    assert np.all(fc_act_res >= 0) and np.all(fc_act_res <= 255), \
                        "Activation assertion failed in fc layer %d (%d)" % (i, np.count_nonzero(fc_act_res > 255))
                if exclude_outliers:
                    range_valid = np.all(fc_act_res <= 255, axis=tuple(np.arange(1, len(fc_act_res.shape))))
                    exclude_cnt = np.count_nonzero(np.logical_not(range_valid))
                    if exclude_cnt > 0:
                        print("Exclude %d examples in fc layer %d" % (exclude_cnt, i))
                        valid_mask = np.logical_and(valid_mask, range_valid)
                if debug:
                    print("reg_fc_%d" % i, fc_act_res[:1].tolist())
            else:
                fc_act_res = fc_bias_res
                if assertion and not exclude_outliers:
                    assert np.all(fc_act_res >= np.iinfo(np.int32).min) and \
                        np.all(fc_act_res <= np.iinfo(np.int32).max), \
                        "Activation assertion failed in fc layer %d" % i
                if exclude_outliers:
                    range_valid = np.all(
                        np.logical_and(fc_act_res >= np.iinfo(np.int32).min, fc_act_res <= np.iinfo(np.int32).max),
                        axis=tuple(np.arange(1, len(fc_act_res.shape))))
                    exclude_cnt = np.count_nonzero(np.logical_not(range_valid))
                    if exclude_cnt > 0:
                        print("Exclude %d examples in fc layer %d" % (exclude_cnt, i))
                        valid_mask = np.logical_and(valid_mask, range_valid)
                if debug:
                    fc_act_res_recover = fc_act_res[:1] * f_fc["bias"].attrs["scale"]
                    print("reg_fc_%d" % i, fc_act_res_recover.tolist())
            if export_fmap is not None:
                fmap_dict["reg_fc_output_%d" % i] = fc_act_res
            last_fmap = fc_act_res
            last_layer_name = "reg_fc_%d" % i
    if export_fmap is not None:
        fmap_dict["outputs"] = last_fmap
    # recover results
    if "activation" in f_net[last_layer_name]:
        zero = f_net[last_layer_name]["activation"].attrs["zero"]
        scale = f_net[last_layer_name]["activation"].attrs["scale"]
    else:
        zero = f_net[last_layer_name]["bias"].attrs["zero"]
        scale = f_net[last_layer_name]["bias"].attrs["scale"]
    result_recover = (last_fmap - zero) * scale
    result_reshape = np.reshape(result_recover, newshape=(sample_cnt, group_size))

    with open(model_result_path, mode="r") as fp:
        result_dict = yaml.load(fp, Loader=yaml.FullLoader)
    mean_slope = result_dict["mean_slope"]
    result_reshape = result_reshape * (1. / (-mean_slope)) * base_cfg["timestep"]

    valid_mask_reshape = np.reshape(valid_mask, newshape=(sample_cnt, group_size))
    valid_mask_comb = np.logical_and(valid_mask_reshape[:, 0], valid_mask_reshape[:, 1])

    final_exclude_cnt = 0
    if exclude_outliers:
        final_exclude_cnt = np.count_nonzero(np.logical_not(valid_mask_comb))
        print("Finally exclude %d sample groups." % final_exclude_cnt)
        result_reshape = result_reshape[valid_mask_comb]

    valid_mask_expand = np.reshape(np.stack((valid_mask_comb, valid_mask_comb), axis=1), newshape=-1)
    if export_fmap is not None:
        original_input_cnt = fmap_dict["inputs"].shape[0]
        if exclude_outliers:
            for key, val in fmap_dict.items():
                fmap_dict[key] = val[valid_mask_expand]
        ef_root = h5py.File(export_fmap, mode="w")
        for key, val in fmap_dict.items():
            ds_export = ef_root.create_dataset(key, shape=val.shape, dtype=val.dtype)
            ds_export[:] = val
        # get valid indexes
        valid_index = np.arange(original_input_cnt, dtype=np.int64)
        if exclude_outliers:
            valid_index = valid_index[valid_mask_expand]
        ds_valid = ef_root.create_dataset("valid_index", shape=valid_index.shape, dtype=valid_index.dtype)
        ds_valid[:] = valid_index
        ds_valid.attrs["count"] = len(valid_index)
        print("Intermediate feature maps have been exported to:", export_fmap)
        ef_root.close()

    print("Simulated inference:")
    print("----------------------------------")
    name = output_spec_names[0].decode("UTF-8")
    if group_size == 2:
        bias, precision = size_both_metric(result_reshape, sample_cnt-final_exclude_cnt)
    else:
        bias, precision = size_many_metric(result_reshape, sample_cnt-final_exclude_cnt)
    print("Slice (%s): bias: %.4f, precision: %.4f" % (name, float(bias), float(precision)))

    f_root.close()
    return result_reshape, valid_mask_expand


def original_inference(config_file, model_save_path, model_result_path, output_keys=("time",), valid=None,
                       debug=False, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]
    if "upd_kwargs" in base_cfg:
        kwargs.update(base_cfg["upd_kwargs"])
    kwargs["config_file"] = config_file

    if "use_toy" in kwargs and kwargs["use_toy"]:
        prepare_data_func = prepare_data_inst_cv
    else:
        prepare_data_func = prepare_data_inst_npz

    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_func)[0]}
    data_handler = prepare_data_func(**data_kwargs)

    model = load_model(model_save_path, compile=False)

    test_data_dict = data_handler.generate_test_val_dataset()
    inputs = test_data_dict["targets"]
    sample_cnt = inputs.shape[0]
    group_size = inputs.shape[1]

    inputs_reshape = np.reshape(inputs, newshape=(sample_cnt * group_size,) + inputs.shape[-2::1])
    final_exclude_cnt = 0
    if valid is not None:
        final_exclude_cnt = np.count_nonzero(np.logical_not(valid)) // 2
        print("Exclude %d sample groups." % final_exclude_cnt)
        inputs_reshape = inputs_reshape[valid]

    outputs = model(inputs_reshape)
    outputs_reshape = np.reshape(outputs, newshape=(-1, group_size))

    with open(model_result_path, mode="r") as fp:
        result_dict = yaml.load(fp, Loader=yaml.FullLoader)
    mean_slope = result_dict["mean_slope"]
    outputs_reshape = outputs_reshape * (1. / (-mean_slope)) * base_cfg["timestep"]

    print("Original inference:")
    print("----------------------------------")
    name = output_keys[0]
    if group_size == 2:
        bias, precision = size_both_metric(outputs_reshape, sample_cnt-final_exclude_cnt)
    else:
        bias, precision = size_many_metric(outputs_reshape, sample_cnt-final_exclude_cnt)
    print("Slice (%s): bias: %.4f, precision: %.4f" % (name, float(bias), float(precision)))

    if debug:
        print("Debug original network (first example):")
        print("----------------------------------")
        extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        features = extractor(inputs[:1])
        features_dict = {k.name: v.numpy()[0].tolist() for k, v in zip(model.layers, features)}
        for k, v in features_dict.items():
            print(k, v)

    return outputs_reshape


if __name__ == "__main__":
    model_export(
        config_file="./conf/default_2ch_internal.yaml",
        param_export_path="./temp/default_2ch/export/default_2ch-toy_seq_model_export.npz",
        hdf5_path="./temp/model_quan_export/default_2ch/export_default_2ch_hyper.hdf5",
        force_override=True
    )
    _, valid_plain = simulated_inference(
        config_file="./conf/default_2ch_internal.yaml",
        hdf5_path="./temp/model_quan_export/default_2ch/export_default_2ch_hyper.hdf5",
        model_result_path="./temp/default_2ch/result/default_2ch-toy_seq_model_res.yaml",
        round_to_even=False,
        debug=True,
        assertion=True,
        exclude_outliers=True,
        export_fmap="./temp/model_quan_export/default_2ch/fmap_default_2ch_hyper.hdf5"
    )
    original_inference(
        config_file="./conf/default_2ch_internal.yaml",
        model_save_path="./temp/default_2ch/model/default_2ch-toy_seq_model",
        model_result_path="./temp/default_2ch/result/default_2ch-toy_seq_model_res.yaml",
        valid=valid_plain,
        debug=False
    )
