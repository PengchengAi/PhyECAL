import inspect
import yaml

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

from src.data_provider import prepare_data_inst_npz, DataHandler
from src.build_model import build_seq_model_base, PhyBindModel
from conf.net_config import AVAILABLE_CONFIGS


def generate_wgt_matrix(length=8):
    x = np.arange(length, dtype=np.float32)
    y = np.ones(shape=length, dtype=np.float32)
    a = np.stack((x, y), axis=0).T
    idn = np.identity(8, dtype=np.float32)
    wgt_matrix = a @ np.linalg.inv(a.T @ a) @ a.T - idn
    return wgt_matrix


def physics_bind_routine(config_file, end_cond_callback=None, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]

    if "upd_kwargs" in base_cfg:
        kwargs.update(base_cfg["upd_kwargs"])

    model_cfg = AVAILABLE_CONFIGS[kwargs["net_cfg_key"]]
    base_model = build_seq_model_base(cfg=model_cfg)

    phy_bind_inst = PhyBindModel(
        base=base_model,
        group=base_cfg["group"],
        wgt_matrix=generate_wgt_matrix(length=base_cfg["group"])
    )

    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_inst_npz)[0]}
    data_kwargs["config_file"] = config_file
    data_handler = prepare_data_inst_npz(**data_kwargs)

    train_data_dict = data_handler.generate_training_dataset()
    if "train_verbose" in base_cfg:
        train_verbose = base_cfg["train_verbose"]
    else:
        train_verbose = "auto"
    # first stage: fit for a certain epochs
    phy_bind_inst.fit(
        train_data_dict["inputs"],
        train_data_dict["labels"],
        batch_size=base_cfg["train_batch_size"],
        epochs=base_cfg["train_epoch"],
        validation_data=None,
        verbose=train_verbose
    )
    # second stage: fit until converge on training set
    end_cond = False
    while end_cond_callback is not None and not end_cond:
        fit_stats_dict = phy_bind_inst.fit(
            train_data_dict["inputs"],
            train_data_dict["labels"],
            batch_size=base_cfg["train_batch_size"],
            epochs=1,
            validation_data=None,
            verbose=0
        )

        if end_cond_callback(fit_stats_dict):
            end_cond = True

    return base_model, phy_bind_inst, data_handler


def base_eval_plot(base_model, data_handler: DataHandler, plot_single=False):
    test_data_dict = data_handler.generate_test_val_dataset()
    targets = test_data_dict["targets"]

    total_samples = targets.shape[0]
    out_result_list = np.zeros(shape=targets.shape[:2], dtype=np.float32)

    for i in range(total_samples):
        out_result = base_model(targets[i, ...])
        out_result_list[i, ...] = np.squeeze(out_result, axis=-1)

    x = np.arange(targets.shape[1], dtype=np.float32)

    fit_result_list = []
    for i in range(total_samples):
        i_res = linregress(x, out_result_list[i, :])
        if plot_single:
            plt.figure()
            plt.title("Fit result of No. %d example (%.4f)" % (i, i_res.rvalue))
            plt.plot(x, out_result_list[i, :], 'o', label='original data')
            plt.plot(x, i_res.intercept + i_res.slope * x, 'r', label='fitted line')
            plt.legend()
            plt.show()
        fit_result_list.append(i_res)

    r_value_list = [item.rvalue for item in fit_result_list]
    plt.figure()
    plt.title("Distribution of R value")
    plt.hist(r_value_list, bins=50)
    plt.show()


def main():
    base_model, _, data_handler = physics_bind_routine(
        config_file="./conf/default_8ch_internal.yaml"
    )
    base_eval_plot(
        base_model=base_model,
        data_handler=data_handler,
        plot_single=True
    )


if __name__ == "__main__":
    main()
