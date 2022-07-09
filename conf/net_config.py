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
        (1, "linear", False, "dense_act_full")
    ),
    "slicing_spec": (
        (0, 1, "time"),
        (1, 2, "energy")
    )
}

full_in640_config = {
    "input_length": 640,
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
        (1, "linear", False, "dense_act_full")
    ),
    "slicing_spec": (
        (0, 1, "time"),
        (1, 2, "energy")
    )
}

compact_in640_config = {
    "input_length": 640,
    "encoder_layer_spec": (
        (2, 16, 8, "relu", False),
        (4, 16, 8, "relu", False)
    ),
    "decoder_layer_spec": (
        (2, 16, 8, "linear", 0.5),
        (1, 16, 8, "relu", False, "conv1d_act_full")
    ),
    "regression_spec": (
        (16, "relu", False),
        (16, "relu", False),
        (1, "linear", False, "dense_act_full")
    ),
    "slicing_spec": (
        (0, 1, "time"),
        (1, 2, "energy")
    )
}

default_in32_config = {
    "input_length": 32,
    "encoder_layer_spec": (
        (8, 4, 2, "relu", False),
        (16, 4, 2, "relu", False),
    ),
    "decoder_layer_spec": (
        (8, 4, 2, "linear", 0.5),
        (1, 4, 2, "relu", False, "conv1d_act_full")
    ),
    "regression_spec": (
        (32, "relu", False),
        (32, "relu", False),
        (1, "linear", False, "dense_act_full")
    ),
    "slicing_spec": (
        (0, 1, "time"),
        (1, 2, "energy")
    )
}

AVAILABLE_CONFIGS = {
    "default_in128": default_in128_config,
    "full_in640": full_in640_config,
    "compact_in640": compact_in640_config,
    "default_in32": default_in32_config
}
