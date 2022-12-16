from ivadomed.loader.generalized_loader_configuration import GeneralizedLoaderConfiguration


def test_generalized_loader_configuration():
    model_dict = {
        "name": "Unet",
        "dropout_rate": 0.3,
        "bn_momentum": 0.1,
        "final_activation": "sigmoid",
        "depth": 3,
    }

    # Build a GeneralizedLoaderConfiguration:
    config: GeneralizedLoaderConfiguration = GeneralizedLoaderConfiguration(
        model_params=model_dict,
    )
    print(config)


