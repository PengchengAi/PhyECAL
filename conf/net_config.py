default_in128_config = {
    "input_length": 128,
    "encoder_layer_spec": (
        (8, 4, 2, "relu", False),
        (16, 4, 2, "relu", False),
        (32, 4, 2, "relu", False),
        (64, 4, 2, "relu", False),
        (64, 4, 2, "relu", False)
    ),
    "decoder_layer_spec": (
        (64, 4, 2, "linear", 0.5),
        (32, 4, 2, "relu", 0.5),
        (16, 4, 2, "relu", False),
        (8, 4, 2, "relu", False),
        (1, 4, 2, "relu", False, "conv1d_act_full")
    ),
    "regression_spec": (
        (64, "relu", False),
        (64, "relu", False),
        (2, "linear", False, "dense_act_full")
    ),
    "slicing_spec": (
        (0, 1, "time"),
        (1, 2, "energy")
    )
}

AVAILABLE_CONFIGS = {
    "default_in128": default_in128_config,
}
