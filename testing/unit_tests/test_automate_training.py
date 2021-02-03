import pytest
from ivadomed.scripts.automate_training import make_config_list, get_param_list, \
    HyperparameterOption
import logging
from unit_tests.t_utils import remove_tmp_dir, create_tmp_dir
logger = logging.getLogger(__name__)

initial_config = {
    'training_parameters': {
        'batch_size': 18,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {
            'initial_lr': 0.001
        }
    },
    'default_model':     {
        'name': 'Unet',
        'dropout_rate': 0.3,
        'bn_momentum': 0.9,
        'depth': 3,
        'is_2d': True
    },
    'log_directory': './tmp/',
    'gpu_ids': [1]
}

expected_config_list_all_combin = [
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=2-gpu_ids=[2]"
    },
    {
        'training_parameters': {
            'batch_size': 2,
            'loss': {'name': 'DiceLoss'},
            'scheduler': {'initial_lr': 0.001}
        },
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=3-gpu_ids=[2]"
    },
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=4-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=2-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=3-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=4-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=2-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=3-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=4-gpu_ids=[2]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=2-gpu_ids=[2]"
    },
    {
        'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=3-gpu_ids=[2]"
    },
    {
        'training_parameters': {
            'batch_size': 64,
            'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
            'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [2],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=4-gpu_ids=[2]"
    },
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=2-gpu_ids=[5]"
    },
    {
        'training_parameters': {
            'batch_size': 2,
            'loss': {'name': 'DiceLoss'},
            'scheduler': {'initial_lr': 0.001}
        },
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=3-gpu_ids=[5]"
    },
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=4-gpu_ids=[5]"
    },
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=2-gpu_ids=[5]"
    },
    {
        'training_parameters': {'batch_size': 2,
                                'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=3-gpu_ids=[5]"
    },
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=4-gpu_ids=[5]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=2-gpu_ids=[5]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=3-gpu_ids=[5]"
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=4-gpu_ids=[5]"
    },
    {
        'training_parameters': {'batch_size': 64,
                                'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=2-gpu_ids=[5]"
    },
    {
        'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=3-gpu_ids=[5]"
    },
    {
        'training_parameters': {
            'batch_size': 64,
            'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
            'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'gpu_ids': [5],
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=4-gpu_ids=[5]"
    }
]


expected_config_list_neither = [
    {
        'training_parameters': {
            'batch_size': 2,
            'loss': {'name': 'DiceLoss'},
            'scheduler': {'initial_lr': 0.001}
        },
        'default_model': {
            'name': 'Unet',
            'dropout_rate': 0.3,
            'bn_momentum': 0.9,
            'depth': 3,
            'is_2d': True
        },
        'log_directory': './tmp/-batch_size=2',
        'gpu_ids': [1]
    },
    {
        'training_parameters': {
            'batch_size': 64,
            'loss': {'name': 'DiceLoss'},
            'scheduler': {'initial_lr': 0.001}
        },
        'default_model': {
            'name': 'Unet',
            'dropout_rate': 0.3,
            'bn_momentum': 0.9,
            'depth': 3,
            'is_2d': True
        },
        'log_directory': './tmp/-batch_size=64',
        'gpu_ids': [1]
    },

    {
        'training_parameters': {
            'batch_size': 18,
            'loss': {'name': 'DiceLoss'},
            'scheduler': {'initial_lr': 0.001}
        },
        'default_model': {
            'name': 'Unet',
            'dropout_rate': 0.3,
            'bn_momentum': 0.9,
            'depth': 3,
            'is_2d': True
            },
        'log_directory': "./tmp/-loss={'name': 'DiceLoss'}",
        'gpu_ids': [1]
    },
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': "./tmp/-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}",
     'gpu_ids': [1]
     },
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 2,
                       'is_2d': True},
     'log_directory': './tmp/-depth=2',
     'gpu_ids': [1]
     },
    {
        'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
        'log_directory': './tmp/-depth=3',
        'gpu_ids': [1]
    },
    {
        'training_parameters': {'batch_size': 18,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': './tmp/-depth=4',
        'gpu_ids': [1]
    },
    {
        'training_parameters': {'batch_size': 18,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': './tmp/-gpu_ids=[2]',
        'gpu_ids': [2]
    },
    {
        'training_parameters': {'batch_size': 18,
                                'loss': {'name': 'DiceLoss'},
                                'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': './tmp/-gpu_ids=[5]',
        'gpu_ids': [5]
    }
]

expected_config_list_multi_param = [
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 2,
                       'is_2d': True},
     'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=2-gpu_ids=[2]",
     'gpu_ids': [2]},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     "log_directory": "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}-depth=3-gpu_ids=[5]",
     'gpu_ids': [5]}

]

expected_param_list = [
    HyperparameterOption('batch_size', {'training_parameters': {'batch_size': 2}}, 2),
    HyperparameterOption('batch_size', {'training_parameters': {'batch_size': 64}}, 64),
    HyperparameterOption('loss', {'training_parameters': {'loss': {'name': 'DiceLoss'}}},
                         {'name': 'DiceLoss'}),
    HyperparameterOption('loss', {'training_parameters': {'loss': {'name': 'FocalLoss',
                                  'gamma': 0.2, 'alpha': 0.5}}},
                         {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}),
    HyperparameterOption('depth', {'default_model': {'depth': 2}}, 2),
    HyperparameterOption('depth', {'default_model': {'depth': 3}}, 3),
    HyperparameterOption('depth', {'default_model': {'depth': 4}}, 4),
    HyperparameterOption('gpu_ids', {'gpu_ids': [2]}, [2]),
    HyperparameterOption('gpu_ids', {'gpu_ids': [5]}, [5])
]


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize("config_hyper", [
    {
        'training_parameters': {
            'batch_size': [2, 64],
            'loss': [
                {'name': 'DiceLoss'},
                {'name': 'FocalLoss', 'gamma': 0.2, 'alpha': 0.5}
             ]
        },
        'default_model': {
            'depth': [2, 3, 4]
        },
        'gpu_ids': [[2], [5]]
    }
])
@pytest.mark.parametrize(
    "expected_param_list",
    [
        pytest.param(
            expected_param_list
        )
    ]
)
def test_get_param_list(config_hyper, expected_param_list):
    param_list = get_param_list(config_hyper, [], [])
    assert param_list == expected_param_list


@pytest.mark.parametrize("initial_config", [initial_config])
@pytest.mark.parametrize(
    "all_combin,multi_params,param_list,expected_config_list",
    [
        pytest.param(
            False, False,
            expected_param_list,
            expected_config_list_neither,
            id="neither"
        ),
        pytest.param(
            True, False,
            expected_param_list,
            expected_config_list_all_combin,
            id="all_combin"
        ),
        pytest.param(
            False, True,
            expected_param_list,
            expected_config_list_multi_param,
            id="multi_param"
        )
    ]
)
def test_make_config_list(initial_config, all_combin, multi_params, param_list,
                          expected_config_list):
    config_list = make_config_list(param_list, initial_config, all_combin, multi_params)
    assert len(config_list) == len(expected_config_list)
    for config_option in config_list:
        print(config_option)
        assert config_option in expected_config_list
    for config_option in expected_config_list:
        assert config_option in config_list


def teardown_function():
    remove_tmp_dir()
