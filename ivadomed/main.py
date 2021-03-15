import json
import os
import argparse
import copy
import joblib
import torch.backends.cudnn as cudnn
import nibabel as nib
import sys
import platform
import multiprocessing
import re

from ivadomed.utils import logger
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

    command_group = parser.add_mutually_exclusive_group(required=False)

    command_group.add_argument("--train", dest='train', action='store_true',
                               help="Perform training on data.")
    command_group.add_argument("--test", dest='test', action='store_true',
                               help="Perform testing on trained model.")
    command_group.add_argument("--segment", dest='segment', action='store_true',
                               help="Perform segmentation on data.")

    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Path to configuration file.")

    # OPTIONAL ARGUMENTS
    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')

    optional_args.add_argument("-pd", "--path-data", dest="path_data", required=False, type=str,
                               nargs="*", help="""Path to data in BIDs format. You may list one
                               or more paths; separate each path with a space, e.g.
                               --path-data some/path/a some/path/b""")
    optional_args.add_argument("-po", "--path-output", required=False, type=str, dest="path_output",
                               help="Path to output directory.")
    optional_args.add_argument('-g', '--gif', required=False, type=int, default=0,
                               help='Number of GIF files to output. Each GIF file corresponds to a 2D slice showing the '
                                    'prediction over epochs (one frame per epoch). The prediction is run on the '
                                    'validation dataset. GIF files are saved in the output path.')
    optional_args.add_argument('-t', '--thr-increment', dest="thr_increment", required=False, type=float,
                               help='A threshold analysis is performed at the end of the training using the trained '
                                    'model and the training+validation sub-datasets to find the optimal binarization '
                                    'threshold. The specified value indicates the increment between 0 and 1 used during '
                                    'the analysis (e.g. 0.1). Plot is saved under "[PATH_OUTPUT]/thr.png" and the '
                                    'optimal threshold in "[PATH_OUTPUT]/config_file.json as "binarize_prediction" '
                                    'parameter.')
    optional_args.add_argument('--resume-training', dest="resume_training", required=False, action='store_true',
                               help='Load a saved model ("checkpoint.pth.tar" in the output directory specified either with flag "--path-output" or via the config file "output_path" argument)  '
                                    'for resume training. This training state is saved everytime a new best model is saved in the output directory specified with flag "--path-output"')
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='Shows function documentation.')

    return parser

def create_path_model(context, model_params, ds_train, path_output, train_onehotencoder):
    path_model = os.path.join(path_output, context["model_name"])
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

def check_multiple_raters(is_train, loader_params):
    if any([isinstance(class_suffix, list) for class_suffix in loader_params["target_suffix"]]):
        print(
            "\nAnnotations from multiple raters will be used during model training, one annotation from one rater "
            "randomly selected at each iteration.\n")
        if not is_train:
            print(
                "\nERROR: Please provide only one annotation per class in 'target_suffix' when not training a model.\n")
            exit()

def film_normalize_data(context, model_params, ds_train, ds_valid, path_output):
    # Normalize metadata before sending to the FiLM network
    results = imed_film.get_film_metadata_models(ds_train=ds_train,
                                                    metadata_type=model_params['metadata'],
                                                    debugging=context["debugging"])
    ds_train, train_onehotencoder, metadata_clustering_models = results
    ds_valid = imed_film.normalize_metadata(ds_valid, metadata_clustering_models, context["debugging"],
                                            model_params['metadata'])
    model_params.update({"film_onehotencoder": train_onehotencoder,
                            "n_metadata": len([ll for l in train_onehotencoder.categories_ for ll in l])})
    joblib.dump(metadata_clustering_models, os.path.join(path_output, "clustering_models.joblib"))
    joblib.dump(train_onehotencoder, os.path.join(path_output + "one_hot_encoder.joblib"))
    
    return model_params, ds_train, ds_valid, train_onehotencoder

def get_dataset(bids_df, loader_params, data_lst, transform_params, cuda_available, device, ds_type):
    ds = imed_loader.load_dataset(bids_df, **{**loader_params, **{'data_list': data_lst,
                                                                  'transforms_params': transform_params,
                                                                  'dataset_type': ds_type}}, device=device,
                                  cuda_available=cuda_available)
    return ds

def save_config_file(context, path_output):
    # Save config file within path_output and path_output/model_name
    # Done after the threshold_analysis to propate this info in the config files
    with open(os.path.join(path_output, "config_file.json"), 'w') as fp:
        json.dump(context, fp, indent=4)
    with open(os.path.join(path_output, context["model_name"], context["model_name"] + ".json"), 'w') as fp:
        json.dump(context, fp, indent=4)

def set_loader_params(context, is_train):
    loader_params = copy.deepcopy(context["loader_parameters"])
    if is_train:
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["training_validation"]
    else:
        loader_params["contrast_params"]["contrast_lst"] = loader_params["contrast_params"]["testing"]
    if "FiLMedUnet" in context and context["FiLMedUnet"]["applied"]:
        loader_params.update({"metadata_type": context["FiLMedUnet"]["metadata"]})
    
    # Load metadata necessary to balance the loader
    if context['training_parameters']['balance_samples']['applied'] and \
            context['training_parameters']['balance_samples']['type'] != 'gt':
        loader_params.update({"metadata_type": context['training_parameters']['balance_samples']['type']})
    return loader_params

def set_model_params(context, loader_params):
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
        object_detection_params.update({"gpu_ids": context['gpu_ids'][0],
                                        "path_output": context['path_output']})
        loader_params.update({"object_detection_params": object_detection_params})

    loader_params.update({"model_params": model_params})

    return model_params, loader_params

def set_output_path(context):
    path_output = copy.deepcopy(context["path_output"])
    if not os.path.isdir(path_output):
        print('Creating output path: {}'.format(path_output))
        os.makedirs(path_output)
    else:
        print('Output path already exists: {}'.format(path_output))
    
    return path_output

def update_film_model_params(context, ds_test, model_params, path_output):
    clustering_path = os.path.join(path_output, "clustering_models.joblib")
    metadata_clustering_models = joblib.load(clustering_path)
    # Model directory
    ohe_path = os.path.join(path_output, context["model_name"], "one_hot_encoder.joblib")
    one_hot_encoder = joblib.load(ohe_path)
    ds_test = imed_film.normalize_metadata(ds_test, metadata_clustering_models, context["debugging"],
                                            model_params['metadata'])
    model_params.update({"film_onehotencoder": one_hot_encoder,
                            "n_metadata": len([ll for l in one_hot_encoder.categories_ for ll in l])})

    return ds_test, model_params

def run_segment_command(context, model_params):

    # BIDSDataframe of all image files
    # Indexing of derivatives is False for command segment
    bids_df = imed_loader_utils.BidsDataframe(context['loader_parameters'], context['path_output'], derivatives=False)

    # Append subjects filenames into a list
    bids_subjects = sorted(bids_df.df['filename'].to_list())

    # Add postprocessing to packaged model
    path_model = os.path.join(context['path_output'], context['model_name'])
    path_model_config = os.path.join(path_model, context['model_name'] + ".json")
    model_config = imed_config_manager.load_json(path_model_config)
    model_config['postprocessing'] = context['postprocessing']
    with open(path_model_config, 'w') as fp:
        json.dump(model_config, fp, indent=4)
    options = None

    # Initialize a list of already seen subject ids for multichannel
    seen_subj_ids = []

    for subject in bids_subjects:
        if context['loader_parameters']['multichannel']:
            # Get subject_id for multichannel
            df_sub = bids_df.df.loc[bids_df.df['filename'] == subject]
            subj_id = re.sub(r'_' + df_sub['suffix'].values[0] + '.*', '', subject)
            if subj_id not in seen_subj_ids:
                # if subj_id has not been seen yet
                fname_img = []
                provided_contrasts = []
                contrasts = context['loader_parameters']['contrast_params']['testing']
                # Keep contrast order
                for c in contrasts:
                    df_tmp = bids_df.df[bids_df.df['filename'].str.contains(subj_id) & bids_df.df['suffix'].str.contains(c)]
                    if ~df_tmp.empty:
                        provided_contrasts.append(c)
                        fname_img.append(df_tmp['path'].values[0])
                seen_subj_ids.append(subj_id)
                if len(fname_img) != len(contrasts):
                    logger.warning("Missing contrast for subject {}. {} were provided but {} are required. Skipping "
                                    "subject.".format(subj_id, provided_contrasts, contrasts))
                    continue
            else:
                # Returns an empty list for subj_id already seen
                fname_img = []
        else:
            fname_img = bids_df.df[bids_df.df['filename'] == subject]['path'].to_list()

        if 'film_layers' in model_params and any(model_params['film_layers']) and model_params['metadata']:
            metadata = bids_df.df[bids_df.df['filename'] == subject][model_params['metadata']].values[0]
            options = {'metadata': metadata}

        if fname_img:
            pred_list, target_list = imed_inference.segment_volume(path_model,
                                                                    fname_images=fname_img,
                                                                    gpu_id=context['gpu_ids'][0],
                                                                    options=options)
            pred_path = os.path.join(context['path_output'], "pred_masks")
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)

            for pred, target in zip(pred_list, target_list):
                filename = subject.split('.')[0] + target + "_pred" + \
                            ".nii.gz"
                nib.save(pred, os.path.join(pred_path, filename))


def run_command(context, n_gif=0, thr_increment=None, resume_training=False):
    """Run main command.

    This function is central in the ivadomed project as training / testing / evaluation commands
    are run via this function. All the process parameters are defined in the config.

    Args:
        context (dict): Dictionary containing all parameters that are needed for a given process. See
            :doc:`configuration_file` for more details.
        n_gif (int): Generates a GIF during training if larger than zero, one frame per epoch for a given slice. The
            parameter indicates the number of 2D slices used to generate GIFs, one GIF per slice. A GIF shows
            predictions of a given slice from the validation sub-dataset. They are saved within the output path.
        thr_increment (float): A threshold analysis is performed at the end of the training using the trained model and
            the training + validation sub-dataset to find the optimal binarization threshold. The specified value
            indicates the increment between 0 and 1 used during the ROC analysis (e.g. 0.1).
        resume_training (bool): Load a saved model ("checkpoint.pth.tar" in the output directory specified with flag "--path-output" or via the config file "output_path" '            This training state is saved everytime a new best model is saved in the log
            argument) for resume training directory.

    Returns:
        float or pandas.DataFrame or None:
            * If "train" command: Returns floats: best loss score for both training and validation.
            * If "test" command: Returns a pandas Dataframe: of metrics computed for each subject of
              the testing sub-dataset and return the prediction metrics before evaluation.
            * If "segment" command: No return value.

    """
    command = copy.deepcopy(context["command"])
    path_output = set_output_path(context)

    # Create a log with the version of the Ivadomed software and the version of the Annexed dataset (if present)
    create_dataset_and_ivadomed_version_log(context)

    cuda_available, device = imed_utils.define_device(context['gpu_ids'][0])

    # BACKWARDS COMPATIBILITY: If bids_path is string, assign to list - Do this here so it propagates to all functions
    context['loader_parameters']['path_data'] = imed_utils.format_path_data(context['loader_parameters']['path_data'])

    # Loader params
    loader_params = set_loader_params(context, command == "train")

    # Get transforms for each subdataset
    transform_train_params, transform_valid_params, transform_test_params = \
        imed_transforms.get_subdatasets_transforms(context["transformation"])

    # MODEL PARAMETERS
    model_params, loader_params = set_model_params(context, loader_params)

    if command == 'segment':
        run_segment_command(context, model_params)
        return
    
    # BIDSDataframe of all image files
    # Indexing of derivatives is True for command train and test
    bids_df = imed_loader_utils.BidsDataframe(loader_params, path_output, derivatives=True)

    # Get subject filenames lists. "segment" command uses all participants of data path, hence no need to split
    train_lst, valid_lst, test_lst = imed_loader_utils.get_subdatasets_subject_files_list(context["split_dataset"],
                                                                                          bids_df.df,
                                                                                          path_output,
                                                                                          context["loader_parameters"]
                                                                                          ['subject_selection'])
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

    # Check if multiple raters
    check_multiple_raters(command != "train", loader_params)

    if command == 'train':
        # Get Validation dataset
        ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device, 'validation')
        
        # Get Training dataset
        ds_train = get_dataset(bids_df, loader_params, train_lst, transform_train_params, cuda_available, device, 'training')
        metric_fns = imed_metrics.get_metric_fns(ds_train.task)

        # If FiLM, normalize data
        if 'film_layers' in model_params and any(model_params['film_layers']):
            model_params, ds_train, ds_valid, train_onehotencoder = \
                film_normalize_data(context, model_params, ds_train, ds_valid, path_output)
        else:
            train_onehotencoder = None

        # Model directory
        create_path_model(context, model_params, ds_train, path_output, train_onehotencoder)

        save_config_file(context, path_output)

        # RUN TRAINING
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = imed_training.train(
            model_params=model_params,
            dataset_train=ds_train,
            dataset_val=ds_valid,
            training_params=context["training_parameters"],
            path_output=path_output,
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
            ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device, 'validation')
        # Get Training dataset with no Data Augmentation
        ds_train = get_dataset(bids_df, loader_params, train_lst, transform_valid_params, cuda_available, device, 'training')

        # Choice of optimisation metric
        metric = "recall_specificity" if model_params["name"] in imed_utils.CLASSIFIER_LIST else "dice"
        # Model path
        model_path = os.path.join(path_output, "best_model.pt")
        # Run analysis
        thr = imed_testing.threshold_analysis(model_path=model_path,
                                              ds_lst=[ds_train, ds_valid],
                                              model_params=model_params,
                                              testing_params=testing_params,
                                              metric=metric,
                                              increment=thr_increment,
                                              fname_out=os.path.join(path_output, "roc.png"),
                                              cuda_available=cuda_available)

        # Update threshold in config file
        context["postprocessing"]["binarize_prediction"] = {"thr": thr}
        save_config_file(context, path_output)

    if command == 'train':
        return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss

    if command == 'test':
        # LOAD DATASET
        ds_test = imed_loader.load_dataset(bids_df, **{**loader_params, **{'data_list': test_lst,
                                                                           'transforms_params': transformation_dict,
                                                                           'dataset_type': 'testing',
                                                                           'requires_undo': True}}, device=device,
                                           cuda_available=cuda_available)

        metric_fns = imed_metrics.get_metric_fns(ds_test.task)

        if 'film_layers' in model_params and any(model_params['film_layers']):
            ds_test, model_params = update_film_model_params(context, ds_test, model_params, path_output)

        # RUN INFERENCE
        pred_metrics = imed_testing.test(model_params=model_params,
                                         dataset_test=ds_test,
                                         testing_params=testing_params,
                                         path_output=path_output,
                                         device=device,
                                         cuda_available=cuda_available,
                                         metric_fns=metric_fns,
                                         postprocessing=context['postprocessing'])

        # RUN EVALUATION
        df_results = imed_evaluation.evaluate(bids_df, path_output=path_output,
                                              target_suffix=loader_params["target_suffix"],
                                              eval_params=context["evaluation_parameters"])
        return df_results, pred_metrics


def create_dataset_and_ivadomed_version_log(context):

    path_data = context['loader_parameters']['path_data']

    ivadomed_version = imed_utils._version_string()
    datasets_version = []

    if isinstance(path_data, str):
        datasets_version = [imed_utils.__get_commit(path_to_git_folder=path_data)]
    elif isinstance(path_data, list):
        for Dataset in path_data:
            datasets_version.append(imed_utils.__get_commit(path_to_git_folder=Dataset))

    log_file = os.path.join(context['path_output'], 'version_info.log')

    try:
        f = open(log_file, "w")
    except OSError as err:
        print("OS error: {0}".format(err))
        raise Exception("Have you selected a log folder, and do you have write permissions for that folder?")

    # IVADOMED
    f.write('IVADOMED TOOLBOX\n----------------\n(' + ivadomed_version + ')')

    # DATASETS
    path_data = imed_utils.format_path_data(path_data)
    f.write('\n\n\nDATASET VERSION\n---------------\n')

    f.write('The following BIDS dataset(s) were used for training.\n')

    for i_dataset in range(len(path_data)):
        if datasets_version[i_dataset] not in ['', '?!?']:
            f.write(str(i_dataset+1) + '. ' + path_data[i_dataset] + ' - Dataset Annex version: ' + datasets_version[i_dataset] + '\n')
        else:
            f.write(str(i_dataset+1) + '. ' + path_data[i_dataset] + ' - Dataset is not Annexed.\n')

    # SYSTEM INFO
    f.write('\n\nSYSTEM INFO\n-------------\n')
    platform_running = sys.platform
    if platform_running.find('darwin') != -1:
        os_running = 'osx'
    elif platform_running.find('linux') != -1:
        os_running = 'linux'
    elif platform_running.find('win32') or platform_running.find('win64'):
        os_running = 'windows'
    else:
        os_running = 'NA'

    f.write('OS: ' + os_running + ' (' + platform.platform() + ')\n')

    # Display number of CPU cores
    f.write('CPU cores: Available: {}\n\n\n\n\n'.format(multiprocessing.cpu_count()))

    # USER INPUTS
    f.write('CONFIG INPUTS\n-------------\n')
    if sys.version_info[0] > 2:
        for k, v in context.items():
            f.write(str(k) + ': ' + str(v) + '\n')  # Making sure all numbers are converted to strings
    else:
        for k, v in context.viewitems():  # Python2
            f.write(str(k) + ': ' + str(v) + '\n')

    f.close()


def run_main():
    imed_utils.init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()

    # Get context from configuration file
    path_config_file = args.config
    context = imed_config_manager.ConfigurationManager(path_config_file).get_config()

    context["command"] = imed_utils.get_command(args, context)
    context["path_output"] = imed_utils.get_path_output(args, context)
    context["loader_parameters"]["path_data"] = imed_utils.get_path_data(args, context)

    # Run command
    run_command(context=context,
                n_gif=args.gif if args.gif is not None else 0,
                thr_increment=args.thr_increment if args.thr_increment else None,
                resume_training=bool(args.resume_training))


if __name__ == "__main__":
    run_main()
