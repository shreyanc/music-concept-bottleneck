batch_size = 8

config_cp_field_shallow_m1 = {
    "name": "config_cp_field_shallow_m1",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [3, 3, 3], "k2s": [1, 3, 3]},
    "stage2": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "stage3": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "block_type": "basic"
}

config_cp_field_shallow_m2 = {
    "name": "config_cp_field_shallow_m2",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [3, 3, 3], "k2s": [1, 3, 3]},
    "stage2": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

config_cp_field_wide = {
    "name": "config_cp_field_shallow_m2",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [5, 3, 3], "k2s": [5, 1, 3]},
    "stage2": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

config_m2_big_kernel = {
    "name": "config_cp_field_shallow_m2",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [6, 6, 6], "k2s": [1, 6, 6]},
    "stage2": {"maxpool": [], "k1s": [3, ], "k2s": [3, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

config_cp_field_shallow_m3 = {
    "name": "config_cp_field_shallow_m3",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [3, 3, 3], "k2s": [1, 3, 3]},
    "stage2": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

config_cp_field_shallow_m4 = {
    "name": "config_cp_field_shallow_m4",
    "input_shape": [batch_size, 1, -1, -1],
    "n_classes": 7,
    "depth": 26,
    "base_channels": 128,
    "n_blocks_per_stage": [3, 1, 1],
    "stage1": {"maxpool": [1, 2], "k1s": [3, 3, 3], "k2s": [1, 3, 1]},
    "stage2": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "stage3": {"maxpool": [], "k1s": [1, ], "k2s": [1, ]},
    "block_type": "basic"
}

def get_cpresnet_config_with_rho(rho, block_type='basic'):
    def get_kernel(idx):
        return 3 if idx <= rho else 1

    k = get_kernel

    config_rho = {
        "name": f"{block_type}_config_rho_{rho}",
        "input_shape": [batch_size, 1, -1, -1],
        "depth": 26,
        "base_channels": 128,
        "n_blocks_per_stage": [3, 3, 3],
        "stage1": {"maxpool": [1, 2], "k1s": [3, k(1), k(3)], "k2s": [1, k(2), k(4)]},
        "stage2": {"maxpool": [2], "k1s": [k(5), k(7), k(9)], "k2s": [k(6), k(8), k(10)]},
        "stage3": {"maxpool": [], "k1s": [k(11), k(13), k(15)], "k2s": [k(12), k(14), k(16)]},
        "block_type": block_type
    }
    return config_rho