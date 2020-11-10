import json
import os
import argparse
import copy
import joblib
import torch.backends.cudnn as cudnn
import nibabel as nib

from bids_neuropoly import bids
from ivadomed import evaluation as imed_evaluation
from ivadomed import config_manager as imed_config_manager
from ivadomed import testing as imed_testing
from ivadomed import training as imed_training
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import metrics as imed_metrics
from ivadomed import inference as imed_inference
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film

cudnn.benchmark = True

# List of not-default available models i.e. different from Unet
MODEL_LIST = ['Modified3DUNet', 'HeMISUnet', 'FiLMedUnet', 'resnet18', 'densenet121', 'Countception']


def get_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # MANDATORY ARGUMENTS
    mandatory_args = parser.add_argument_group('MANDATORY ARGUMENTS')
    mandatory_args.add_argument("-c", "--config", required=True, type=str,
                                help="Path to configuration file.")

    # OPTIONAL ARGUMENTS
    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')
    optional_args.add_argument('-g', '--gif', required=False, type=int, default=0,
                               help='Generates a GIF of during training, one frame per epoch for a given slice.'
                                    ' The parameter indicates the number of 2D slices used to generate GIFs, one GIF '
                                    'per slice. A GIF shows predictions of a given slice from the validation '
                                    'sub-dataset. They are saved within the log directory.')
    optional_args.add_argument('-t', '--thr-increment', dest="thr_increment", required=False, type=float,
                               help='A threshold analysis is performed at the end of the training using the trained '
                                    'model and the training+validation sub-datasets to find the optimal binarization '
                                    'threshold. The specified value indicates the increment between 0 and 1 used during '
                                    'the analysis (e.g. 0.1). Plot is saved under "log_directory/thr.png" and the '
                                    'optimal threshold in "log_directory/config_file.json as "binarize_prediction" '
                                    'parameter.')
    optional_args.add_argument('--resume-training', dest="resume_training", required=False, action='store_true',
                               help='Load a saved model ("checkpoint.pth.tar" in the log_directory) for resume '
                                    'training. This training state is saved everytime a new best model is saved in the'
                                    'log directory.')
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='Shows function documentation.')

    return parser


def run_command(context, n_gif=0, thr_increment=None, resume_training=False):
    """Run main command.

    This function is central in the ivadomed project as training / testing / evaluation commands are run via this
    function. All the process parameters are defined in the config.

    Args:
        context (dict): Dictionary containing all parameters that are needed for a given process. See
            :doc:`configuration_file` for more details.
        n_gif (int): Generates a GIF during training if larger than zero, one frame per epoch for a given slice. The
            parameter indicates the number of 2D slices used to generate GIFs, one GIF per slice. A GIF shows
            predictions of a given slice from the validation sub-dataset. They are saved within the log directory.
        thr_increment (float): A threshold analysis is performed at the end of the training using the trained model and
            the training + validation sub-dataset to find the optimal binarization threshold. The specified value
            indicates the increment between 0 and 1 used during the ROC analysis (e.g. 0.1).
        resume_training (bool): Load a saved model ("checkpoint.pth.tar" in the log_directory) for resume training.
            This training state is saved everytime a new best model is saved in the log
            directory.

    Returns:
        Float or pandas Dataframe:
        If "train" command: Returns floats: best loss score for both training and validation.
        If "test" command: Returns a pandas Dataframe: of metrics computed for each subject of the testing
            sub dataset and return the prediction metrics before evaluation.
        If "segment" command: No return value.
    """
    command = copy.deepcopy(context["command"])
    log_directory = copy.deepcopy(context["log_directory"])
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
    loader_params = copy.deepcopy(context["loader_parameters"])
    if command == "train":
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    else:
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["testing"]
    if "FiLMedUnet" in context and context["FiLMedUnet"]["applied"]:
        loader_params.update({"metadata_type": context["FiLMedUnet"]["metadata"]})

    # Get transforms for each subdataset
    transform_train_params, transform_valid_params, transform_test_params = \
        imed_transforms.get_subdatasets_transforms(context["transformation"])

    # MODEL PARAMETERS
    model_params = copy.deepcopy(context["default_model"])
    model_params["folder_name"] = copy.deepcopy(context["model_name"])
    model_context_list = [model_name for model_name in MODEL_LIST
                          if model_name in context and context[model_name]["applied"]]
    if len(model_context_list) == 1:
        model_params["name"] = model_context_list[0]
        model_params.update(context[model_context_list[0]])
    elif 'Modified3DUNet' in model_context_list and 'FiLMedUnet' in model_context_list and len(model_context_list) == 2:
        model_params["name"] = 'Modified3DUNet'
        for i in range(len(model_context_list)):
            model_params.update(context[model_context_list[i]])
    elif len(model_context_list) > 1:
        print('ERROR: Several models are selected in the configuration file: {}.'
              'Please select only one (i.e. only one where: "applied": true).'.format(model_context_list))
        exit()

    model_params['is_2d'] = False if "Modified3DUNet" in model_params['name'] else model_params['is_2d']
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

    # TESTING PARAMS
    # Aleatoric uncertainty
    if context['uncertainty']['aleatoric'] and context['uncertainty']['n_it'] > 0:
        transformation_dict = transform_train_params
    else:
        transformation_dict = transform_test_params
    undo_transforms = imed_transforms.UndoCompose(imed_transforms.Compose(transformation_dict, requires_undo=True))
    testing_params = copy.deepcopy(context["training_parameters"])
    testing_params.update({'uncertainty': context["uncertainty"]})
    testing_params.update({'target_suffix': loader_params["target_suffix"], 'undo_transforms': undo_transforms,
                           'slice_axis': loader_params['slice_axis']})
    if command == "train":
        imed_utils.display_selected_transfoms(transform_train_params, dataset_type=["training"])
        imed_utils.display_selected_transfoms(transform_valid_params, dataset_type=["validation"])
    elif command == "test":
        imed_utils.display_selected_transfoms(transformation_dict, dataset_type=["testing"])

    if command == 'train':
        # LOAD DATASET
        # Get Validation dataset
        ds_valid = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': valid_lst, 'transforms_params': transform_valid_params,
                                                  'dataset_type': 'validation'}}, device=device,
                                            cuda_available=cuda_available)
        # Get Training dataset
        ds_train = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': train_lst, 'transforms_params': transform_train_params,
                                                  'dataset_type': 'training'}}, device=device,
                                            cuda_available=cuda_available)

        metric_fns = imed_metrics.get_metric_fns(ds_train.task)

        # If FiLM, normalize data
        if 'film_layers' in model_params and any(model_params['film_layers']):
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
            if 'film_layers' in model_params and any(model_params['film_layers']):
                joblib.dump(train_onehotencoder, os.path.join(path_model, "one_hot_encoder.joblib"))
                if 'metadata_dict' in ds_train[0]['input_metadata'][0]:
                    metadata_dict = ds_train[0]['input_metadata'][0]['metadata_dict']
                    joblib.dump(metadata_dict, os.path.join(path_model, "metadata_dict.joblib"))

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
            n_gif=n_gif,
            resume_training=resume_training,
            debugging=context["debugging"])

    if thr_increment:
        # LOAD DATASET
        if command != 'train':  # If command == train, then ds_valid already load
            # Get Validation dataset
            ds_valid = imed_loader.load_dataset(**{**loader_params,
                                                   **{'data_list': valid_lst, 'transforms_params': transform_valid_params,
                                                      'dataset_type': 'validation'}}, device=device,
                                                cuda_available=cuda_available)
        # Get Training dataset with no Data Augmentation
        ds_train = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': train_lst, 'transforms_params': transform_valid_params,
                                                  'dataset_type': 'training'}}, device=device,
                                            cuda_available=cuda_available)

        # Choice of optimisation metric
        metric = "recall_specificity" if model_params["name"] in imed_utils.CLASSIFIER_LIST else "dice"
        # Model path
        model_path = os.path.join(log_directory, "best_model.pt")
        # Run analysis
        thr = imed_testing.threshold_analysis(model_path=model_path,
                                              ds_lst=[ds_train, ds_valid],
                                              model_params=model_params,
                                              testing_params=testing_params,
                                              metric=metric,
                                              increment=thr_increment,
                                              fname_out=os.path.join(log_directory, "roc.png"),
                                              cuda_available=cuda_available)

        # Update threshold in config file
        context["postprocessing"]["binarize_prediction"] = {"thr": thr}

    if command == 'train':
        # Save config file within log_directory and log_directory/model_name
        # Done after the threshold_analysis to propate this info in the config files
        with open(os.path.join(log_directory, "config_file.json"), 'w') as fp:
            json.dump(context, fp, indent=4)
        with open(os.path.join(log_directory, context["model_name"], context["model_name"] + ".json"), 'w') as fp:
            json.dump(context, fp, indent=4)

        return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss

    if command == 'test':
        # LOAD DATASET
        ds_test = imed_loader.load_dataset(**{**loader_params, **{'data_list': test_lst,
                                                                  'transforms_params': transformation_dict,
                                                                  'dataset_type': 'testing',
                                                                  'requires_undo': True}}, device=device,
                                                                  cuda_available=cuda_available)

        metric_fns = imed_metrics.get_metric_fns(ds_test.task)

        if 'film_layers' in model_params and any(model_params['film_layers']):
            clustering_path = os.path.join(log_directory, "clustering_models.joblib")
            metadata_clustering_models = joblib.load(clustering_path)
            ohe_path = os.path.join(log_directory, "one_hot_encoder.joblib")
            one_hot_encoder = joblib.load(ohe_path)
            ds_test = imed_film.normalize_metadata(ds_test, metadata_clustering_models, context["debugging"],
                                                   model_params['metadata'])
            model_params.update({"film_onehotencoder": one_hot_encoder,
                                 "n_metadata": len([ll for l in one_hot_encoder.categories_ for ll in l])})

        # RUN INFERENCE
        pred_metrics = imed_testing.test(model_params=model_params,
                                         dataset_test=ds_test,
                                         testing_params=testing_params,
                                         log_directory=log_directory,
                                         device=device,
                                         cuda_available=cuda_available,
                                         metric_fns=metric_fns,
                                         postprocessing=context['postprocessing'])

        # RUN EVALUATION
        df_results = imed_evaluation.evaluate(bids_path=loader_params['bids_path'],
                                              log_directory=log_directory,
                                              target_suffix=loader_params["target_suffix"],
                                              eval_params=context["evaluation_parameters"])
        return df_results, pred_metrics

    if command == 'segment':
        bids_ds = bids.BIDS(context["loader_parameters"]["bids_path"])
        df = bids_ds.participants.content
        subj_lst = df['participant_id'].tolist()
        bids_subjects = [s for s in bids_ds.get_subjects() if s.record["subject_id"] in subj_lst]

        # Add postprocessing to packaged model
        path_model = os.path.join(context['log_directory'], context['model_name'])
        path_model_config = os.path.join(path_model, context['model_name'] + ".json")
        model_config = imed_config_manager.load_json(path_model_config)
        model_config['postprocessing'] = context['postprocessing']
        with open(path_model_config, 'w') as fp:
            json.dump(model_config, fp, indent=4)

        options = None
        for subject in bids_subjects:
            fname_img = subject.record["absolute_path"]
            if 'film_layers' in model_params and any(model_params['film_layers']) and model_params['metadata']:
                subj_id = subject.record['subject_id']
                metadata = df[df['participant_id'] == subj_id][model_params['metadata']].values[0]
                options = {'metadata': metadata}
            pred = imed_inference.segment_volume(path_model,
                                                 fname_image=fname_img,
                                                 gpu_number=context['gpu'],
                                                 options=options)
            pred_path = os.path.join(context['log_directory'], "pred_masks")
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            filename = subject.record['subject_id'] + "_" + subject.record['modality'] + "_pred" + ".nii.gz"
            nib.save(pred, os.path.join(pred_path, filename))


def run_main():
    imed_utils.init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()

    # Get context from configuration file
    path_config_file = args.config
    context = imed_config_manager.ConfigurationManager(path_config_file).get_config()

    # Run command
    run_command(context=context,
                n_gif=args.gif if args.gif is not None else 0,
                thr_increment=args.thr_increment if args.thr_increment else None,
                resume_training=bool(args.resume_training))


if __name__ == "__main__":
    run_main()
