import pytest

from ivadomed.scripts.automate_training import make_category, make_config_list
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
    'log_directory': './tmp/'
}

expected_items_simple = [
    {
        'name': 'Unet',
        'dropout_rate': 0.3,
        'bn_momentum': 0.9,
        'depth': 2,
        'is_2d': True
    },
    {
        'name': 'Unet',
        'dropout_rate': 0.3,
        'bn_momentum': 0.9,
        'depth': 3,
        'is_2d': True
    },
    {
        'name': 'Unet',
        'dropout_rate': 0.3,
        'bn_momentum': 0.9,
        'depth': 4,
        'is_2d': True
    }]

expected_names_simple = ['-depth=2', '-depth=3', '-depth=4']

expected_items_complex_neither = [
    {
        'batch_size': 2,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 32,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 64,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 18,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 18,
        'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 18,
        'loss': {'name': 'GeneralizedDiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    }
]

expected_names_complex_neither = [
    '-batch_size=2',
    '-batch_size=32',
    '-batch_size=64',
    "-loss={'name': 'DiceLoss'}",
    "-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}",
    "-loss={'name': 'GeneralizedDiceLoss'}"
]

expected_items_complex_all_combin = [
    {
        'batch_size': 2,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 2,
        'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 2,
        'loss': {'name': 'GeneralizedDiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 32,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 32,
        'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 32,
        'loss': {'name': 'GeneralizedDiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 64,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 64,
        'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
        'scheduler': {'initial_lr': 0.001}
    },
    {
        'batch_size': 64,
        'loss': {'name': 'GeneralizedDiceLoss'},
        'scheduler': {'initial_lr': 0.001}
    }
]

expected_names_complex_all_combin = [
    "-batch_size=2-loss={'name': 'DiceLoss'}",
    "-batch_size=2-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}",
    "-batch_size=2-loss={'name': 'GeneralizedDiceLoss'}",
    "-batch_size=32-loss={'name': 'DiceLoss'}",
    "-batch_size=32-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}",
    "-batch_size=32-loss={'name': 'GeneralizedDiceLoss'}",
    "-batch_size=64-loss={'name': 'DiceLoss'}",
    "-batch_size=64-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}",
    "-batch_size=64-loss={'name': 'GeneralizedDiceLoss'}"
]

param_dict = {
    'training_parameters': expected_items_complex_all_combin,
    'default_model': expected_items_simple
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
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=2"},
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
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'DiceLoss'}-depth=4"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=2"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=3"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=4"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'GeneralizedDiceLoss'}-depth=2"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'GeneralizedDiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 2,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=2-loss={'name': 'GeneralizedDiceLoss'}-depth=4"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'DiceLoss'}-depth=2"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'DiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'DiceLoss'}-depth=4"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=2"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=3"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=4"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'GeneralizedDiceLoss'}-depth=2"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'GeneralizedDiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 32,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=32-loss={'name': 'GeneralizedDiceLoss'}-depth=4"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=2"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'DiceLoss'}-depth=4"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=2"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=3"},
    {
        'training_parameters': {
            'batch_size': 64,
            'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
            'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}-depth=4"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 2,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'GeneralizedDiceLoss'}-depth=2"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 3,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'GeneralizedDiceLoss'}-depth=3"},
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
        'default_model': {'name': 'Unet',
                          'dropout_rate': 0.3,
                          'bn_momentum': 0.9,
                          'depth': 4,
                          'is_2d': True},
        'log_directory': "./tmp/-batch_size=64-loss={'name': 'GeneralizedDiceLoss'}-depth=4"}
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
        'log_directory': './tmp/-batch_size=2'
    },
    {
        'training_parameters': {
            'batch_size': 32,
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
        'log_directory': './tmp/-batch_size=32'
    },
    {'training_parameters': {'batch_size': 64,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': './tmp/-batch_size=64'},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': "./tmp/-loss={'name': 'DiceLoss'}"},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': "./tmp/-loss={'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}}"},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'GeneralizedDiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': "./tmp/-loss={'name': 'GeneralizedDiceLoss'}"},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 2,
                       'is_2d': True},
     'log_directory': './tmp/-depth=2'},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 3,
                       'is_2d': True},
     'log_directory': './tmp/-depth=3'},
    {'training_parameters': {'batch_size': 18,
                             'loss': {'name': 'DiceLoss'},
                             'scheduler': {'initial_lr': 0.001}},
     'default_model': {'name': 'Unet',
                       'dropout_rate': 0.3,
                       'bn_momentum': 0.9,
                       'depth': 4,
                       'is_2d': True},
     'log_directory': './tmp/-depth=4'}
]


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize("category_init", [
    {
        'name': 'Unet',
        'dropout_rate': 0.3,
        'bn_momentum': 0.9,
        'depth': 3,
        'is_2d': True}
    ])
@pytest.mark.parametrize("category_hyper", [{'depth': [2, 3, 4]}])
@pytest.mark.parametrize(
    "is_all_combin,multiple_params,expected_items,expected_names",
    [
        pytest.param(
            True, True, expected_items_simple, expected_names_simple, id="is_all_combin_multi_param"
        ),
        pytest.param(
            False, False, expected_items_simple, expected_names_simple, id="neither"
        ),
        pytest.param(
            True, False, expected_items_simple, expected_names_simple, id="is_all_combin"
        )
    ]
)
def test_make_category_simple(category_init, category_hyper, is_all_combin, multiple_params,
                              expected_items, expected_names):
    items, names = make_category(category_init, category_hyper, is_all_combin,
                                 multiple_params)
    assert items == expected_items
    assert names == expected_names


@pytest.mark.parametrize("category_init", [
    {
        'batch_size': 18,
        'loss': {'name': 'DiceLoss'},
        'scheduler': {
            'initial_lr': 0.001
        }
    }
    ])
@pytest.mark.parametrize("category_hyper", [
    {
        'batch_size': [2, 32, 64],
        'loss': [
            {'name': 'DiceLoss'},
            {'name': 'FocalLoss', 'params': {'gamma': 0.2, 'alpha': 0.5}},
            {'name': 'GeneralizedDiceLoss'}
         ]
    }
])
@pytest.mark.parametrize(
    "is_all_combin,multiple_params,expected_items,expected_names",
    [
        pytest.param(
            True, True, expected_items_complex_all_combin,
            expected_names_complex_all_combin, id="is_all_combin_multi_param"
        ),
        pytest.param(
            False, False, expected_items_complex_neither,
            expected_names_complex_neither, id="neither"
        ),
        pytest.param(
            True, False, expected_items_complex_all_combin,
            expected_names_complex_all_combin, id="is_all_combin"
        )
    ]
)
def test_make_category_complex(category_init, category_hyper, is_all_combin, multiple_params,
                               expected_items, expected_names):
    items, names = make_category(category_init, category_hyper, is_all_combin,
                                 multiple_params)
    assert items == expected_items
    assert names == expected_names


@pytest.mark.parametrize("initial_config", [initial_config])
@pytest.mark.parametrize(
    "is_all_combin,multiple_params,param_dict,names_dict,expected_config_list",
    [
        pytest.param(
            True, True,
            {
                'training_parameters': expected_items_complex_all_combin,
                'default_model': expected_items_simple
            },
            {
                'training_parameters': expected_names_complex_all_combin,
                'default_model': expected_names_simple
            },
            expected_config_list_all_combin,
            id="is_all_combin_multi_param"
        ),
        pytest.param(
            False, False,
            {
                'training_parameters': expected_items_complex_neither,
                'default_model': expected_items_simple
            },
            {
                'training_parameters': expected_names_complex_neither,
                'default_model': expected_names_simple
            },
            expected_config_list_neither,
            id="neither"
        ),
        pytest.param(
            True, False,
            {
                'training_parameters': expected_items_complex_all_combin,
                'default_model': expected_items_simple
            },
            {
                'training_parameters': expected_names_complex_all_combin,
                'default_model': expected_names_simple
            },
            expected_config_list_all_combin,
            id="is_all_combin"
        ),
    ]
)
def test_make_config_list(initial_config, is_all_combin, multiple_params, param_dict, names_dict,
                          expected_config_list):
    config_list = make_config_list(names_dict, param_dict, initial_config, is_all_combin,
                                   multiple_params)
    assert config_list == expected_config_list


def teardown_function():
    remove_tmp_dir()
