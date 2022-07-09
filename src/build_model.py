import os
import yaml
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras import Sequential

from src.util import update


def build_seq_model_base(cfg, name="seq_model", l2_norm=None):
    # generate input layer
    input_length = cfg["input_length"]
    input_layer: list[layers.Layer] = [layers.InputLayer(input_shape=(input_length, 1))]
    # generate encoder layers
    encoder_layer_spec = cfg["encoder_layer_spec"]
    encoder_layers: list[layers.Layer] = []

    for ind, (filters, kernel_size, strides, activation, dropout, *args) in enumerate(encoder_layer_spec):
        if l2_norm is not None:
            k_reg = regularizers.l2(l2_norm)
            b_reg = regularizers.l2(l2_norm)
        else:
            k_reg = None
            b_reg = None
        enc_layer = layers.Conv1D(filters, kernel_size, strides=strides, padding="same", name="enc_conv_%d" % ind,
                                  activation=activation, kernel_regularizer=k_reg, bias_regularizer=b_reg)
        encoder_layers.append(enc_layer)
        if dropout:
            encoder_layers.append(layers.Dropout(rate=dropout, name="enc_conv_%d_dropout" % ind))
    # generate regression layers
    regression_spec = cfg["regression_spec"]
    regression_layers: list[layers.Layer] = [layers.Flatten(name="reg_input_flat")]
    for ind, (weights, activation, dropout, *args) in enumerate(regression_spec):
        if l2_norm is not None:
            if ind == len(regression_spec) - 1:
                k_reg = None
                b_reg = None
            else:
                k_reg = regularizers.l2(l2_norm)
                b_reg = regularizers.l2(l2_norm)
        else:
            k_reg = None
            b_reg = None
        reg_layer = layers.Dense(weights, name="reg_fc_%d" % ind, activation=activation, kernel_regularizer=k_reg,
                                 bias_regularizer=b_reg)
        regression_layers.append(reg_layer)
        if dropout:
            regression_layers.append(layers.Dropout(rate=dropout, name="reg_fc_%d_dropout" % ind))

    # generate model
    seq_layers = input_layer + encoder_layers + regression_layers
    model = Sequential(layers=seq_layers, name=name)
    return model


class PhyBindModel(Model, ABC):
    def __init__(self, base, group, wgt_matrix, **kwargs):
        super(PhyBindModel, self).__init__(**kwargs)
        self.base = base
        self.group = group
        self.wgt_matrix = wgt_matrix
        self.total_loss_tracker = metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        assert len(x.shape) == 4 and x.shape[1] == self.group, "Shape mismatches."

        with tf.GradientTape() as tape:
            x_input = tf.reshape(x, shape=(x.shape[0] * x.shape[1],) + x.shape[2:], name="x_input_reshape")
            y_output = self.base(x_input, training=True)
            y_pred = tf.reshape(y_output, shape=y.shape, name="y_output_reshape")
            y_diff = y_pred - y
            y_wgt = tf.reshape(
                tf.tile(self.wgt_matrix, multiples=[y.shape[0], 1]),
                shape=(y.shape[0], self.group, self.group)
            )
            residuals = tf.matmul(y_wgt, y_diff, name="result_matmul")
            total_loss = tf.reduce_mean(residuals ** 2)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }


class ToyPhyBindModel(Model, ABC):
    def __init__(self, base, random_start, sample_rate, sample_pts, **kwargs):
        super(ToyPhyBindModel, self).__init__(**kwargs)
        self.base = base
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.sample_pts = sample_pts
        self.total_loss_tracker = metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        assert len(x.shape) == 4 and x.shape[1] == 2, "Shape mismatches."

        with tf.GradientTape() as tape:
            # generate sampling tensor
            start_ind = tf.random.categorical(
                tf.math.log([[1./self.random_start]*self.random_start]*x.shape[0]), 1,
                dtype=tf.int32
            ) + 1
            rnd_shift = tf.random.categorical(
                tf.math.log([[0.5, 0.5]]*x.shape[0]), 1,
                dtype=tf.int32
            ) * 2 - 1
            shift_comb = tf.concat(
                (rnd_shift, tf.zeros_like(rnd_shift)),
                axis=1
            )
            start_ind = tf.reshape(start_ind + shift_comb, shape=(x.shape[0], 2, 1))
            range_ind = tf.reshape(
                tf.tile(
                    tf.range(0, self.sample_rate * self.sample_pts, delta=self.sample_rate, dtype=tf.int32),
                    multiples=[x.shape[0] * 2]
                ),
                shape=(x.shape[0], 2, self.sample_pts)
            )
            sel_ind = start_ind + range_ind
            x_gather = tf.gather(
                x,
                sel_ind,
                axis=2,
                batch_dims=2,
            )
            # infer with network
            x_input = tf.reshape(x_gather, shape=(x.shape[0] * 2, self.sample_pts, 1), name="x_input_reshape")
            y_output = self.base(x_input, training=True)
            y_pred = y_output[0::2] - y_output[1::2]
            y = -rnd_shift
            y_diff = y_pred - tf.cast(y, dtype=tf.float32)
            total_loss = tf.reduce_mean(y_diff ** 2)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }


def compile_model(model: Model, net_compile_key="adam"):
    model.compile(optimizer=net_compile_key)


def save_seq_model(model: Model, config_file, upd_dict=None, data_key="bind", name="seq_model"):
    with open(config_file, mode="r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    if upd_dict is not None:
        cfg = update(cfg, upd_dict)
        print("Configuration has been updated with the dictionary:", upd_dict)

    model_save_dir = cfg["global"]["model_save_dir"]
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_name = "%s-%s_%s" % (cfg["supp"]["save_prefix"], data_key, name)
    model_path = os.path.join(model_save_dir, model_name)
    model.save(filepath=model_path)
    print("Model %s has been saved to: %s" % (name, model_path))


def test_model():
    base_inputs = Input(shape=(32, 1))
    x = layers.Conv1D(32, 4, activation="relu", strides=2, padding="same")(base_inputs)
    x = layers.Conv1D(64, 4, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(1, activation=None)(x)
    base = Model(base_inputs, x, name="encoder")
    base.summary()

    a = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32).T
    idn = np.identity(8, dtype=np.float32)
    wgt_matrix = a @ np.linalg.inv(a.T @ a) @ a.T - idn

    phy_bind_model = PhyBindModel(base=base, group=8, wgt_matrix=wgt_matrix)

    x_fake = np.random.random(size=(128, 8, 32, 1)).astype(np.float32)
    y_fake = np.random.random(size=(128, 8, 1)).astype(np.float32)

    print(base.trainable_weights[0][0][0])
    phy_bind_model.compile(optimizer=optimizers.Adam())
    phy_bind_model.fit(x=x_fake, y=y_fake, batch_size=16, epochs=40, verbose=0)
    print(base.trainable_weights[0][0][0])


def test_model_2():
    base_inputs = Input(shape=(32, 1))
    x = layers.Conv1D(32, 4, activation="relu", strides=2, padding="same")(base_inputs)
    x = layers.Conv1D(64, 4, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(1, activation=None)(x)
    base = Model(base_inputs, x, name="encoder")
    base.summary()

    phy_bind_model = ToyPhyBindModel(base=base, random_start=10, sample_rate=4, sample_pts=32)

    x_fake = np.random.random(size=(128, 2, 160, 1)).astype(np.float32)
    y_fake = np.random.random(size=(128, 2, 1)).astype(np.float32)

    print(base.trainable_weights[0][0][0])
    phy_bind_model.compile(optimizer=optimizers.Adam())
    phy_bind_model.fit(x=x_fake, y=y_fake, batch_size=16, epochs=40, verbose=0)
    print(base.trainable_weights[0][0][0])


if __name__ == "__main__":
    test_model()
    test_model_2()
