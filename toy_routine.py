import yaml
import inspect

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm

from conf.net_config import AVAILABLE_CONFIGS
from src.build_model import build_seq_model, ToyPhyBindModel, compile_model
from src.build_model import save_seq_model, export_seq_model, save_eval_results
from src.quan_aux import quantize_scope, quantize_apply
from src.data_provider import prepare_data_inst_cv


def couple_routine(config_file, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    base_cfg = cfg["baseline"]

    if "upd_kwargs" in base_cfg:
        kwargs.update(base_cfg["upd_kwargs"])
    kwargs["config_file"] = config_file

    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_inst_cv)[0]}
    save_model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(save_seq_model)[0]}
    export_model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(export_seq_model)[0]}
    save_results_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(save_eval_results)[0]}

    data_handler = prepare_data_inst_cv(**data_kwargs)
    train_dataset = data_handler.generate_training_dataset()
    test_dataset = data_handler.generate_test_val_dataset()

    # prepare data
    training_data_fit = train_dataset["inputs"]
    fake_label_data = train_dataset["labels"]
    training_data = np.squeeze(training_data_fit, axis=-1)
    test_data = np.squeeze(test_dataset["targets"], axis=-1)

    # prepare model
    model_cfg = AVAILABLE_CONFIGS[kwargs["net_cfg_key"]]
    base_model, q_cfg_dict = build_seq_model(cfg=model_cfg)
    base_model.summary()
    phy_bind_inst = ToyPhyBindModel(
        base=base_model,
        random_start=base_cfg["random_start"],
        sample_rate=base_cfg["sample_rate"],
        sample_pts=base_cfg["sample_pts"]
    )
    compile_model(model=phy_bind_inst, net_compile_key=kwargs["net_compile_key"])

    # train model and slope correction
    phy_bind_inst.fit(
        x=training_data_fit,
        y=fake_label_data,
        batch_size=base_cfg["train_batch_size"],
        epochs=base_cfg["train_epoch"],
        verbose=1
    )
    training_data_res = np.reshape(training_data, newshape=(-1, training_data.shape[-1], 1))
    infer_result_arr = np.zeros(shape=(training_data_res.shape[0], base_cfg["slope_range"], 1), dtype=np.float32)
    for i in range(base_cfg["slope_range"]):
        training_data_slice = training_data_res[:, i::base_cfg["slope_range"], :]
        infer_result = base_model(training_data_slice)
        infer_result_arr[:, i, :] = infer_result
    x = np.linspace(0, 1, base_cfg["slope_range"], endpoint=False)
    if kwargs["verbose"] >= 1:
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
    test_diff = test_diff * (1. / (-mean_slope)) * base_cfg["timestep"]
    mean = np.mean(test_diff)
    std = np.std(test_diff)
    if kwargs["verbose"] >= 1:
        plt.figure()
        plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(test_diff), float(mean), float(std)))
        _, bins, _ = plt.hist(test_diff.numpy(), density=True)
        x = np.linspace(bins[0], bins[-1], 101, endpoint=True)
        plt.plot(x, norm.pdf(x, loc=mean, scale=std), "r--")
        plt.show()

    print("Baseline evaluation results:")
    print("---------------------------------------")
    print("entries = %d, mean slope = %.3f, mean = %.3f, std. = %.3f" % (
        len(test_diff), float(mean_slope), float(mean), float(std)))

    baseline_metrics_dict = {
        "entries": len(test_diff),
        "mean_slope": float(mean_slope),
        "mean": float(mean),
        "std": float(std)
    }

    if base_cfg["save_model"]:
        save_model_kwargs["model"] = base_model
        if "save_name_override" in base_cfg:
            save_model_kwargs["name"] = base_cfg["save_name_override"]
        save_seq_model(**save_model_kwargs)

    if base_cfg["export_model"]:
        export_model_kwargs["model"] = base_model
        if "export_name_override" in base_cfg:
            export_model_kwargs["name"] = base_cfg["export_name_override"]
        export_seq_model(**export_model_kwargs)

    if base_cfg["save_results"]:
        save_results_kwargs["result_dict"] = baseline_metrics_dict
        if "result_name_override" in base_cfg:
            save_results_kwargs["name"] = base_cfg["result_name_override"]
        save_eval_results(**save_results_kwargs)

    return base_model, data_handler, q_cfg_dict


def couple_routine_quan(config_file, base_model=None, data_handler=None, q_cfg_dict=None, **kwargs):
    if base_model is None or data_handler is None or q_cfg_dict is None:
        print("Do not pass in base model or data handler. Train from scratch.")
        base_model, data_handler, q_cfg_dict = couple_routine(config_file=config_file, **kwargs)

    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    quan_cfg = cfg["quantize"]
    if "upd_kwargs" in quan_cfg:
        kwargs.update(quan_cfg["upd_kwargs"])
    kwargs["config_file"] = config_file

    save_model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(save_seq_model)[0]}
    export_model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(export_seq_model)[0]}
    save_results_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(save_eval_results)[0]}

    train_dataset = data_handler.generate_training_dataset()
    test_dataset = data_handler.generate_test_val_dataset()

    # prepare data
    training_data_fit = train_dataset["inputs"]
    fake_label_data = train_dataset["labels"]
    training_data = np.squeeze(training_data_fit, axis=-1)
    test_data = np.squeeze(test_dataset["targets"], axis=-1)

    # quantization-aware training
    # q_aware stands for quantization aware
    with quantize_scope(q_cfg_dict):
        q_aware_model = quantize_apply(base_model)
    q_aware_model.summary()
    phy_bind_inst = ToyPhyBindModel(
        base=q_aware_model,
        random_start=quan_cfg["random_start"],
        sample_rate=quan_cfg["sample_rate"],
        sample_pts=quan_cfg["sample_pts"]
    )
    compile_model(model=phy_bind_inst, net_compile_key=kwargs["net_compile_key"])
    phy_bind_inst.fit(
        x=training_data_fit,
        y=fake_label_data,
        batch_size=quan_cfg["train_batch_size"],
        epochs=quan_cfg["train_epoch"],
        verbose=1
    )
    # slope correction
    training_data_res = np.reshape(training_data, newshape=(-1, training_data.shape[-1], 1))
    infer_result_arr = np.zeros(shape=(training_data_res.shape[0], quan_cfg["slope_range"], 1), dtype=np.float32)
    for i in range(quan_cfg["slope_range"]):
        training_data_slice = training_data_res[:, i::quan_cfg["slope_range"], :]
        infer_result = q_aware_model(training_data_slice)
        infer_result_arr[:, i, :] = infer_result
    x = np.linspace(0, 1, quan_cfg["slope_range"], endpoint=False)
    if kwargs["verbose"] >= 1:
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
    test_diff = test_diff * (1. / (-mean_slope)) * quan_cfg["timestep"]
    mean = np.mean(test_diff)
    std = np.std(test_diff)
    if kwargs["verbose"] >= 1:
        plt.figure()
        plt.title("entries = %d, mean = %.3f, std. = %.3f" % (len(test_diff), float(mean), float(std)))
        _, bins, _ = plt.hist(test_diff.numpy(), density=True)
        x = np.linspace(bins[0], bins[-1], 101, endpoint=True)
        plt.plot(x, norm.pdf(x, loc=mean, scale=std), "r--")
        plt.show()

    print("Quantized network evaluation results:")
    print("---------------------------------------")
    print("entries = %d, mean slope = %.3f, mean = %.3f, std. = %.3f" % (
        len(test_diff), float(mean_slope), float(mean), float(std)))

    q_aware_metrics_dict = {
        "entries": len(test_diff),
        "mean_slope": float(mean_slope),
        "mean": float(mean),
        "std": float(std)
    }

    if quan_cfg["save_model"]:
        save_model_kwargs["model"] = q_aware_model
        if "save_name_override" in quan_cfg:
            save_model_kwargs["name"] = quan_cfg["save_name_override"]
        save_seq_model(**save_model_kwargs)

    if quan_cfg["export_model"]:
        export_model_kwargs["model"] = q_aware_model
        if "export_name_override" in quan_cfg:
            export_model_kwargs["name"] = quan_cfg["export_name_override"]
        export_seq_model(**export_model_kwargs)

    if quan_cfg["save_results"]:
        save_results_kwargs["result_dict"] = q_aware_metrics_dict
        if "result_name_override" in quan_cfg:
            save_results_kwargs["name"] = quan_cfg["result_name_override"]
        save_eval_results(**save_results_kwargs)

    return q_aware_model, data_handler, q_cfg_dict


if __name__ == "__main__":
    couple_routine_quan(
        config_file="./conf/default_2ch_internal.yaml",
        fold_index=0
    )
