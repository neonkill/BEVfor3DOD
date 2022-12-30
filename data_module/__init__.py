# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


from data_module.dataset import nuscenes_dataset
# from . import nuscenes_dataset_generated


MODULES = {
    'nuscenes_dataset': nuscenes_dataset,
    # 'nuscenes_generated': nuscenes_dataset_generated,
}


def get_dataset_module_by_name(name):
    return MODULES[name]
