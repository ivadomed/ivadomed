import json
import os
import sys

import joblib
import torch.backends.cudnn as cudnn

from ivadomed import evaluation as imed_evaluation
from ivadomed import metrics as imed_metrics
from ivadomed import testing as imed_testing
from ivadomed import training as imed_training
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film

cudnn.benchmark = True

# List of not-default available models i.e. different from Unet

MODEL_LIST = ['UNet3D', 'HeMISUnet', 'FiLMedUnet', 'NAME_CLASSIFIER_1', 'Countception']


def run_main(config=None):
    """Run main command.

    This function is central in the ivadomed project as training / testing / evaluation commands are run via this
    function. All the process parameters are defined in the config.

    Args:
        config (dict): Dictionary containing all parameters that are needed for a given process. See
            :doc:`configuration_file` for more details.

    Returns:
        If "train" command: Returns floats: best loss score for both training and validation.
        If "test" command: Returns dict: of averaged metrics computed on the testing sub dataset.
        If "eval" command: Returns a pandas Dataframe: of metrics computed for each subject of the testing sub dataset.
    """
    # Necessary when calling run_main through python code instead of command-line
    if config is None:
        if len(sys.argv) != 2:
            print("\nERROR: Please indicate the path of your configuration file, "
                  "e.g. ivadomed ivadomed/config/config.json\n")
            return

        path_config_file = sys.argv[1]
        if not os.path.isfile(path_config_file) or not path_config_file.endswith('.json'):
            print("\nERROR: The provided configuration file path (.json) is invalid: {}\n".format(path_config_file))
            return

        with open(path_config_file, "r") as fhandle:
            context = json.load(fhandle)
    else:
        context = config

    command = context["command"]
    log_directory = context["log_directory"]
    if not os.path.isdir(log_directory):
        print('Creating log directory: {}'.format(log_directory))
        os.makedirs(log_directory)
    else:
        print('Log directory already exists: {}'.format(log_directory))

    # Define device
    cuda_available, device = imed_utils.define_device(context['gpu'])

    # Get subject lists
    train_lst, valid_lst, test_lst = imed_loader_utils.get_subdatasets_subjects_list(context["split_dataset"],
                                                                                     context['loader_parameters']
                                                                                     ['bids_path'],
                                                                                     log_directory)

    # Loader params
    loader_params = context["loader_parameters"]
    if command == "train":
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    else:
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["testing"]
    if "FiLMedUnet" in context and context["FiLMedUnet"]["applied"]:
        loader_params.update({"metadata_type": context["FiLMedUnet"]["metadata"]})

    # Get transforms for each subdataset
    transform_train_params, transform_valid_params, transform_test_params = \
        imed_transforms.get_subdatasets_transforms(context["transformation"])
    if command == "train":
        imed_utils.display_selected_transfoms(transform_train_params, dataset_type="training")
        imed_utils.display_selected_transfoms(transform_valid_params, dataset_type="validation")
    elif command == "test":
        imed_utils.display_selected_transfoms(transform_test_params, dataset_type="testing")

    # METRICS
    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.multi_class_dice_score,
                  imed_metrics.hausdorff_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    # MODEL PARAMETERS
    model_params = context["default_model"]
    model_params["folder_name"] = context["model_name"]
    model_context_list = [model_name for model_name in MODEL_LIST
                          if model_name in context and context[model_name]["applied"]]
    if len(model_context_list) == 1:
        model_params["name"] = model_context_list[0]
        model_params.update(context[model_context_list[0]])
    elif len(model_context_list) > 1:
        print('ERROR: Several models are selected in the configuration file: {}.'
              'Please select only one (i.e. only one where: "applied": true).'.format(model_context_list))
        exit()
    # Get in_channel from contrast_lst
    if loader_params["multichannel"]:
        model_params["in_channel"] = len(loader_params["contrast_params"]["contrast_lst"])
    else:
        model_params["in_channel"] = 1
    # Get out_channel from target_suffix
    model_params["out_channel"] = len(loader_params["target_suffix"])
    # If multi-class output, then add background class
    if model_params["out_channel"] > 1:
        model_params.update({"out_channel": model_params["out_channel"] + 1})
    # Display for spec' check
    imed_utils.display_selected_model_spec(params=model_params)
    # Update loader params
    if 'object_detection_params' in context:
        object_detection_params = context['object_detection_params']
        object_detection_params.update({"gpu": context['gpu'],
                                        "log_directory": context['log_directory']})
        loader_params.update({"object_detection_params": object_detection_params})

    loader_params.update({"model_params": model_params})

    if command == 'train':

        # LOAD DATASET
        # Get Validation dataset
        ds_valid = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': valid_lst, 'transforms_params': transform_valid_params,
                                                  'dataset_type': 'validation'}})
        # Get Training dataset
        ds_train = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': train_lst, 'transforms_params': transform_train_params,
                                                  'dataset_type': 'training'}})

        # If FiLM, normalize data
        if model_params["name"] == "FiLMedUnet":
            # Normalize metadata before sending to the FiLM network
            results = imed_film.get_film_metadata_models(ds_train=ds_train,
                                                         metadata_type=model_params['metadata'],
                                                         debugging=context["debugging"])
            ds_train, train_onehotencoder, metadata_clustering_models = results
            ds_valid = imed_film.normalize_metadata(ds_valid, metadata_clustering_models, context["debugging"],
                                                    model_params['metadata'])
            model_params.update({"film_onehotencoder": train_onehotencoder,
                                 "n_metadata": len([ll for l in train_onehotencoder.categories_ for ll in l])})
            joblib.dump(metadata_clustering_models, "./" + log_directory + "/clustering_models.joblib")
            joblib.dump(train_onehotencoder, "./" + log_directory + "/one_hot_encoder.joblib")

        # Model directory
        path_model = os.path.join(log_directory, context["model_name"])
        if not os.path.isdir(path_model):
            print('Creating model directory: {}'.format(path_model))
            os.makedirs(path_model)
        else:
            print('Model directory already exists: {}'.format(path_model))

        # RUN TRAINING
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = imed_training.train(
            model_params=model_params,
            dataset_train=ds_train,
            dataset_val=ds_valid,
            training_params=context["training_parameters"],
            log_directory=log_directory,
            device=device,
            cuda_available=cuda_available,
            metric_fns=metric_fns,
            debugging=context["debugging"])

        # Save config file within log_directory and log_directory/model_name
        with open(os.path.join(log_directory, "config_file.json"), 'w') as fp:
            json.dump(context, fp, indent=4)
        with open(os.path.join(log_directory, context["model_name"], context["model_name"] + ".json"), 'w') as fp:
            json.dump(context, fp, indent=4)

        return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss

    elif command == 'test':
        # LOAD DATASET
        # Aleatoric uncertainty
        if context['testing_parameters']['uncertainty']['aleatoric'] and \
                context['testing_parameters']['uncertainty']['n_it'] > 0:
            transformation_dict = transform_valid_params
        else:
            transformation_dict = transform_test_params

        # UNDO TRANSFORMS
        undo_transforms = imed_transforms.UndoCompose(imed_transforms.Compose(transformation_dict))

        # Get Testing dataset
        ds_test = imed_loader.load_dataset(**{**loader_params, **{'data_list': test_lst,
                                                                  'transforms_params': transformation_dict,
                                                                  'dataset_type': 'testing',
                                                                  'requires_undo': True}})

        if model_params["name"] == "FiLMedUnet":
            clustering_path = os.path.join(log_directory, "clustering_models.joblib")
            metadata_clustering_models = joblib.load(clustering_path)
            ohe_path = os.path.join(log_directory, "one_hot_encoder.joblib")
            one_hot_encoder = joblib.load(ohe_path)
            ds_test = imed_film.normalize_metadata(ds_test, metadata_clustering_models, context["debugging"],
                                                   model_params['metadata'])
            model_params.update({"film_onehotencoder": one_hot_encoder,
                                 "n_metadata": len([ll for l in one_hot_encoder.categories_ for ll in l])})

        # RUN INFERENCE
        testing_params = context["testing_parameters"]
        testing_params.update(context["training_parameters"])
        testing_params.update({'target_suffix': loader_params["target_suffix"], 'undo_transforms': undo_transforms,
                               'slice_axis': loader_params['slice_axis']})
        metrics_dict = imed_testing.test(model_params=model_params,
                                         dataset_test=ds_test,
                                         testing_params=testing_params,
                                         log_directory=log_directory,
                                         device=device,
                                         cuda_available=cuda_available,
                                         metric_fns=metric_fns)
        return metrics_dict

    elif command == 'eval':
        # PREDICTION FOLDER
        path_preds = os.path.join(log_directory, 'pred_masks')
        # If the prediction folder does not exist, run Inference first
        if not os.path.isdir(path_preds):
            print('\nRun Inference\n')
            context["command"] = "test"
            run_main(context)

        # RUN EVALUATION
        df_results = imed_evaluation.evaluate(bids_path=loader_params['bids_path'],
                                              log_directory=log_directory,
                                              path_preds=path_preds,
                                              target_suffix=loader_params["target_suffix"],
                                              eval_params=context["evaluation_parameters"])
        return df_results


if __name__ == "__main__":
    run_main()
