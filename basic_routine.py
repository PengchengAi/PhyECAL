import os
import inspect
import yaml

import numpy as np
from keras.models import load_model
from keras.callbacks import Callback
from scipy.stats import linregress
import matplotlib.pyplot as plt

from src.data_provider import prepare_data_inst_npz, DataHandler
from src.build_model import build_seq_model_base, PhyBindModel, compile_model, save_seq_model
from conf.net_config import AVAILABLE_CONFIGS


def generate_wgt_matrix(length=8):
    x = np.arange(length, dtype=np.float32)
    y = np.ones(shape=length, dtype=np.float32)
    a = np.stack((x, y), axis=0).T
    idn = np.identity(8, dtype=np.float32)
    wgt_matrix = a @ np.linalg.inv(a.T @ a) @ a.T - idn
    return wgt_matrix


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


class DrawSampleCallback(Callback):
    def __init__(self, test_data, save_dir):
        super(DrawSampleCallback, self).__init__()
        self.test_data = test_data
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 50 == 0:
            base_model = self.model.base

            total_samples = self.test_data.shape[0]
            assert total_samples == 8, "Unexpected total samples."
            out_result_list = np.zeros(shape=self.test_data.shape[:2], dtype=np.float32)

            for i in range(total_samples):
                out_result = base_model(self.test_data[i, ...])
                out_result_list[i, ...] = np.squeeze(out_result, axis=-1)

            fig = plt.figure()
            _, h = fig.get_size_inches()
            fig.set_size_inches(h * 4 * 0.8, h * 2 * 0.8)
            axes = fig.subplots(2, 4)
            plt.suptitle("Fit result at %d epochs" % (epoch+1))

            x = np.arange(self.test_data.shape[1], dtype=np.float32)

            for i in range(total_samples):
                row = i // 4
                col = i % 4
                i_res = linregress(x, out_result_list[i, :])
                axes[row, col].plot(x, out_result_list[i, :], 'o', label='original data')
                axes[row, col].plot(x, i_res.intercept + i_res.slope * x, 'r', label='fitted line')

            save_name = "figure_on_epoch_%d.jpg" % (epoch+1)
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path)
            plt.close("all")

            print("Save figure to %s" % save_path)


def physics_bind_routine(config_file, end_cond_callback=None, **kwargs):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]

    if "upd_kwargs" in base_cfg:
        kwargs.update(base_cfg["upd_kwargs"])

    kwargs["config_file"] = config_file
    data_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(prepare_data_inst_npz)[0]}
    save_model_kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(save_seq_model)[0]}

    model_cfg = AVAILABLE_CONFIGS[kwargs["net_cfg_key"]]
    base_model = build_seq_model_base(cfg=model_cfg)

    phy_bind_inst = PhyBindModel(
        base=base_model,
        group=base_cfg["group"],
        wgt_matrix=generate_wgt_matrix(length=base_cfg["group"])
    )
    compile_model(model=phy_bind_inst, net_compile_key=kwargs["net_compile_key"])

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
        verbose=train_verbose,
        callbacks=[
            DrawSampleCallback(
                test_data=train_data_dict["targets"][:8],
                save_dir=os.path.join(cfg["global"]["result_save_dir"], "figs")
            )
        ]
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

    if base_cfg["save_model"]:
        save_model_kwargs["model"] = base_model
        if "save_name_override" in base_cfg:
            save_model_kwargs["name"] = base_cfg["save_name_override"]
        save_seq_model(**save_model_kwargs)

    return base_model, phy_bind_inst, data_handler


def base_eval_plot(config_file, base_model, data_handler: DataHandler, plot_single=False):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    base_cfg = cfg["baseline"]

    test_data_dict = data_handler.generate_test_val_dataset()
    targets = test_data_dict["targets"]

    if isinstance(base_model, str):
        base_model = load_model(base_model, compile=False)

    total_samples = targets.shape[0]
    out_result_list = np.zeros(shape=targets.shape[:2], dtype=np.float32)

    for i in range(total_samples):
        out_result = base_model(targets[i, ...])
        out_result_list[i, ...] = np.squeeze(out_result, axis=-1)

    if base_cfg["save_results"]:
        result_save_dir = cfg["global"]["result_save_dir"]
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)
        result_name = "%s-result.npz" % cfg["supp"]["save_prefix"]
        result_path = os.path.join(result_save_dir, result_name)
        np.savez(
            result_path,
            result=out_result_list
        )
        print("Evaluation results of model have been saved to: %s" % result_path)

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


def main(config_file, base_model=None, plot_single=True):
    if base_model is None:
        base_model, _, data_handler = physics_bind_routine(
            config_file=config_file
        )
    else:
        data_handler = get_data_handler(
            config_file=config_file
        )

    base_eval_plot(
        config_file=config_file,
        base_model=base_model,
        data_handler=data_handler,
        plot_single=plot_single
    )


if __name__ == "__main__":
    main(
        config_file="./conf/default_8ch_internal.yaml",
        base_model=None,
        plot_single=True
    )
