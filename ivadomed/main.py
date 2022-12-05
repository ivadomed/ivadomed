import json
import argparse
import copy
import joblib
import torch.backends.cudnn as cudnn
import nibabel as nib
import sys
import platform
import multiprocessing
import re
from typing import Tuple

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import evaluation as imed_evaluation
from ivadomed import config_manager as imed_config_manager
from ivadomed import testing as imed_testing
from ivadomed import training as imed_training
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import metrics as imed_metrics
from ivadomed import inference as imed_inference
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film
from ivadomed.keywords import ConfigKW, ModelParamsKW, LoaderParamsKW, ContrastParamsKW, BalanceSamplesKW, \
    TrainingParamsKW, ObjectDetectionParamsKW, UncertaintyKW, PostprocessingKW, BinarizeProdictionKW, MetricsKW, \
    MetadataKW, OptionKW, SplitDatasetKW, CommandKW, DataloaderKW, DatasetTypeKW
from loguru import logger
from pathlib import Path

from ivadomed.loader.consolidation import ConsolidatedDataset
from ivadomed.loader.generalized_loader_configuration import GeneralizedLoaderConfiguration

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
    optional_args.add_argument('--no-patch', dest="no_patch", action='store_true', required=False,
                               help='2D patches are not used while segmenting with models trained with patches '
                               '(command "--segment" only). The "--no-patch" argument supersedes the "--overlap-2D" argument. '
                               ' This option may not be suitable with large images depending on computer RAM capacity.')
    optional_args.add_argument('--overlap-2d', dest="overlap_2d", required=False, type=int, nargs="+",
                                help='Custom overlap for 2D patches while segmenting (command "--segment" only). '
                                'Example: "--overlap-2d 48 48" for an overlap of 48 pixels between patches in X and Y respectively. '
                                'Default model overlap is used otherwise.')
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='Shows function documentation.')

    return parser


def create_path_model(context, model_params, ds_train, path_output, train_onehotencoder):
    path_model = Path(path_output, context[ConfigKW.MODEL_NAME])
    if not path_model.is_dir():
        logger.info(f'Creating model directory: {path_model}')
        path_model.mkdir(parents=True)
        if ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS]):
            joblib.dump(train_onehotencoder, path_model.joinpath("one_hot_encoder.joblib"))
            if MetadataKW.METADATA_DICT in ds_train[0][MetadataKW.INPUT_METADATA][0]:
                metadata_dict = ds_train[0][MetadataKW.INPUT_METADATA][0][MetadataKW.METADATA_DICT]
                joblib.dump(metadata_dict, path_model.joinpath("metadata_dict.joblib"))

    else:
        logger.info(f'Model directory already exists: {path_model}')


def check_multiple_raters(is_train, loader_params):
    if any([isinstance(class_suffix, list) for class_suffix in loader_params[LoaderParamsKW.TARGET_SUFFIX]]):
        logger.info(
            "Annotations from multiple raters will be used during model training, one annotation from one rater "
            "randomly selected at each iteration.\n")
        if not is_train:
            logger.error(
                "Please provide only one annotation per class in 'target_suffix' when not training a model.\n")
            sys.exit()


def film_normalize_data(context, model_params, ds_train, ds_valid, path_output):
    # Normalize metadata before sending to the FiLM network
    results = imed_film.get_film_metadata_models(ds_train=ds_train,
                                                 metadata_type=model_params[ModelParamsKW.METADATA],
                                                 debugging=context[ConfigKW.DEBUGGING])
    ds_train, train_onehotencoder, metadata_clustering_models = results
    ds_valid = imed_film.normalize_metadata(ds_valid, metadata_clustering_models, context[ConfigKW.DEBUGGING],
                                            model_params[ModelParamsKW.METADATA])
    model_params.update({ModelParamsKW.FILM_ONEHOTENCODER: train_onehotencoder,
                         ModelParamsKW.N_METADATA: len([ll for l in train_onehotencoder.categories_ for ll in l])})
    joblib.dump(metadata_clustering_models, Path(path_output, "clustering_models.joblib"))
    joblib.dump(train_onehotencoder, Path(path_output + "one_hot_encoder.joblib"))

    return model_params, ds_train, ds_valid, train_onehotencoder


def get_dataset(bids_df, loader_params, data_lst, transform_params, cuda_available, device, ds_type):
    """
    Obtain either the BIDS2D or BIDS3D type dataset.
    """
    ds = imed_loader.load_dataset(bids_df, **{**loader_params, **{'data_list': data_lst,
                                                                  'transforms_params': transform_params,
                                                                  'dataset_type': ds_type}}, device=device,
                                  cuda_available=cuda_available)
    return ds


def save_config_file(context, path_output):
    # Save config file within path_output and path_output/model_name
    # Done after the threshold_analysis to propate this info in the config files
    with Path(path_output, "config_file.json").open(mode='w') as fp:
        json.dump(context, fp, indent=4)
    with Path(path_output, context[ConfigKW.MODEL_NAME], context[ConfigKW.MODEL_NAME] + ".json").open(mode='w') as fp:
        json.dump(context, fp, indent=4)


def update_loader_params(context: dict, is_train: bool, loader_version: str):
    """
    Update the loader parameters dictionary
    """
    # Copy Get the loader parameter from context first.
    loader_params: dict = copy.deepcopy(context[ConfigKW.LOADER_PARAMETERS])

    # Only BIDS Loader has the contrast param key
    if loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:
        if is_train:
            loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LST] = \
                loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TRAINING_VALIDATION]
        else:
            loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LST] =\
                loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING]

    if ConfigKW.FILMED_UNET in context and context[ConfigKW.FILMED_UNET][ModelParamsKW.APPLIED]:
        loader_params.update({LoaderParamsKW.METADATA_TYPE: context[ConfigKW.FILMED_UNET][ModelParamsKW.METADATA]})

    # Load metadata necessary to balance the loader
    if context[ConfigKW.TRAINING_PARAMETERS][TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.APPLIED] and \
            context[ConfigKW.TRAINING_PARAMETERS][TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.TYPE] != 'gt':
        loader_params.update({LoaderParamsKW.METADATA_TYPE:
                                  context[ConfigKW.TRAINING_PARAMETERS][TrainingParamsKW.BALANCE_SAMPLES][BalanceSamplesKW.TYPE]})
    return loader_params


def set_model_params(context: dict, loader_params: dict) -> Tuple[dict, dict]:
    """
    Update the model parameters using context.
    Args:
        context:
        loader_params:

    Returns:

    """

    # Normally use Default Model which is UNET
    model_params: dict = copy.deepcopy(context[ConfigKW.DEFAULT_MODEL])

    # Set the folder name to be the model name
    model_params[ModelParamsKW.FOLDER_NAME] = copy.deepcopy(context[ConfigKW.MODEL_NAME])

    # Generate the model Context List by checking against all support model lists and see if they needs to be applied
    # Per the configuration file
    model_context_list = []
    for model_name in MODEL_LIST:
        if model_name in context and context.get(model_name).get(ModelParamsKW.APPLIED):
            model_context_list.append(model_name)

    if len(model_context_list) == 1:
        model_params[ModelParamsKW.NAME] = model_context_list[0]
        model_params.update(context[model_context_list[0]])
    elif ConfigKW.MODIFIED_3D_UNET in model_context_list and ConfigKW.FILMED_UNET in model_context_list \
            and len(model_context_list) == 2:
        model_params[ModelParamsKW.NAME] = ConfigKW.MODIFIED_3D_UNET
        for i in range(len(model_context_list)):
            model_params.update(context[model_context_list[i]])
    elif len(model_context_list) > 1:
        logger.error(f'ERROR: Several models are selected in the configuration file: {model_context_list}.'
              'Please select only one (i.e. only one where: "applied": true).')
        exit()

    if ConfigKW.MODIFIED_3D_UNET in model_params[ModelParamsKW.NAME]:
        model_params[ModelParamsKW.IS_2D] = False
    else:
        model_params[ModelParamsKW.IS_2D]

    # Get in_channel from contrast_lst
    if loader_params[LoaderParamsKW.MULTICHANNEL]:
        model_params[ModelParamsKW.IN_CHANNEL] = \
            len(loader_params[LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.CONTRAST_LST])
    # If new Loader V2, this needs to be determined based on the AllDataSet Spec
    elif context.get(DataloaderKW.DATASET_GROUPS):
        model_params[ModelParamsKW.IN_CHANNEL] = context.get(DataloaderKW.EXPECTED_INPUT)

    # Old Loader V1
    if loader_params[LoaderParamsKW.TARGET_SUFFIX]:
        # Get out_channel from target_suffix
        model_params[ModelParamsKW.OUT_CHANNEL] = len(loader_params[LoaderParamsKW.TARGET_SUFFIX])
    # New loader V2, this needs to be determined based on the AllDataSet Spec
    elif context.get(DataloaderKW.DATASET_GROUPS):
        model_params[ModelParamsKW.OUT_CHANNEL] = context.get(DataloaderKW.EXPECTED_GT)

    # If multi-class output, then add background class
    if model_params[ModelParamsKW.OUT_CHANNEL] > 1:
        model_params.update(
            {
                ModelParamsKW.OUT_CHANNEL: model_params[ModelParamsKW.OUT_CHANNEL] + 1
            }
        )

    # Display for spec check
    imed_utils.display_selected_model_spec(params=model_params)

    # Update loader params
    if ConfigKW.OBJECT_DETECTION_PARAMS in context:
        object_detection_params = context[ConfigKW.OBJECT_DETECTION_PARAMS]
        object_detection_params.update({ObjectDetectionParamsKW.GPU_IDS: context[ConfigKW.GPU_IDS][0],
                                        ObjectDetectionParamsKW.PATH_OUTPUT: context[ConfigKW.PATH_OUTPUT]})
        loader_params.update({ConfigKW.OBJECT_DETECTION_PARAMS: object_detection_params})

    loader_params.update({LoaderParamsKW.MODEL_PARAMS: model_params})

    return model_params, loader_params

def set_output_path(context: dict) -> str:
    """
    Ensure output path exists
    """
    path_output = copy.deepcopy(context.get(ConfigKW.PATH_OUTPUT))
    if not Path(path_output).is_dir():
        logger.info(f'Creating output path: {path_output}')
        Path(path_output).mkdir(parents=True)
    else:
        logger.info(f'Output path already exists: {path_output}')

    return path_output

def update_film_model_params(context, ds_test, model_params, path_output):
    clustering_path = Path(path_output, "clustering_models.joblib")
    metadata_clustering_models = joblib.load(clustering_path)
    # Model directory
    ohe_path = Path(path_output, context[ConfigKW.MODEL_NAME], "one_hot_encoder.joblib")
    one_hot_encoder = joblib.load(ohe_path)
    ds_test = imed_film.normalize_metadata(ds_test, metadata_clustering_models, context[ConfigKW.DEBUGGING],
                                           model_params[ModelParamsKW.METADATA])
    model_params.update({ModelParamsKW.FILM_ONEHOTENCODER: one_hot_encoder,
                         ModelParamsKW.N_METADATA: len([ll for l in one_hot_encoder.categories_ for ll in l])})

    return ds_test, model_params


def run_segment_command(context, model_params, no_patch, overlap_2d):
    # BIDSDataframe of all image files
    # Indexing of derivatives is False for command segment
    # split_method is unused for command segment
    bids_df = BidsDataframe(
        context.get(ConfigKW.LOADER_PARAMETERS),
        context.get(ConfigKW.PATH_OUTPUT),
        derivatives=False,
        split_method=None
    )

    # Append subjects filenames into a list
    bids_subjects = sorted(bids_df.df.get('filename').to_list())

    # Add postprocessing to packaged model
    path_model = Path(context[ConfigKW.PATH_OUTPUT], context[ConfigKW.MODEL_NAME])
    path_model_config = Path(path_model, context[ConfigKW.MODEL_NAME] + ".json")
    model_config = imed_config_manager.load_json(str(path_model_config))
    model_config[ConfigKW.POSTPROCESSING] = context.get(ConfigKW.POSTPROCESSING)
    with path_model_config.open(mode='w') as fp:
        json.dump(model_config, fp, indent=4)
    options = {}

    # Initialize a list of already seen subject ids for multichannel
    seen_subj_ids = []

    for subject in bids_subjects:
        if context.get(ConfigKW.LOADER_PARAMETERS).get(LoaderParamsKW.MULTICHANNEL):
            # Get subject_id for multichannel
            df_sub = bids_df.df.loc[bids_df.df['filename'] == subject]
            subj_id = re.sub(r'_' + df_sub['suffix'].values[0] + '.*', '', subject)
            if subj_id not in seen_subj_ids:
                # if subj_id has not been seen yet
                fname_img = []
                provided_contrasts = []
                contrasts = context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING]
                # Keep contrast order
                for c in contrasts:
                    df_tmp = bids_df.df[
                        bids_df.df['filename'].str.contains(subj_id) & bids_df.df['suffix'].str.contains(c)]
                    if ~df_tmp.empty:
                        provided_contrasts.append(c)
                        fname_img.append(df_tmp['path'].values[0])
                seen_subj_ids.append(subj_id)
                if len(fname_img) != len(contrasts):
                    logger.warning(f"Missing contrast for subject {subj_id}. {provided_contrasts} were provided but "
                                   f"{contrasts} are required. Skipping subject.")
                    continue
            else:
                # Returns an empty list for subj_id already seen
                fname_img = []
        else:
            fname_img = bids_df.df[bids_df.df['filename'] == subject]['path'].to_list()

        # Add film metadata to options for segment_volume
        if ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS]) \
                and model_params[ModelParamsKW.METADATA]:
            metadata = bids_df.df[bids_df.df['filename'] == subject][model_params[ModelParamsKW.METADATA]].values[0]
            options[OptionKW.METADATA] = metadata

        # Add microscopy pixel size and pixel size units metadata to options for segment_volume
        if MetadataKW.PIXEL_SIZE in bids_df.df.columns:
            options[OptionKW.PIXEL_SIZE] = \
                bids_df.df.loc[bids_df.df['filename'] == subject][MetadataKW.PIXEL_SIZE].values[0]
        if MetadataKW.PIXEL_SIZE_UNITS in bids_df.df.columns:
            options[OptionKW.PIXEL_SIZE_UNITS] = \
                bids_df.df.loc[bids_df.df['filename'] == subject][MetadataKW.PIXEL_SIZE_UNITS].values[0]

        # Add 'no_patch' and 'overlap-2d' argument to options
        if no_patch:
            options[OptionKW.NO_PATCH] = no_patch
        if overlap_2d:
            options[OptionKW.OVERLAP_2D] = overlap_2d

        if fname_img:
            pred_list, target_list = imed_inference.segment_volume(str(path_model),
                                                                   fname_images=fname_img,
                                                                   gpu_id=context[ConfigKW.GPU_IDS][0],
                                                                   options=options)
            pred_path = Path(context[ConfigKW.PATH_OUTPUT], "pred_masks")
            if not pred_path.exists():
                pred_path.mkdir(parents=True)

            # Reformat target list to include class index and be compatible with multiple raters
            target_list = ["_class-%d" % i for i in range(len(target_list))]

            for pred, target in zip(pred_list, target_list):
                filename = subject.split('.')[0] + target + "_pred" + ".nii.gz"
                nib.save(pred, Path(pred_path, filename))

            # For Microscopy PNG/TIF files (TODO: implement OMETIFF behavior)
            extension = imed_loader_utils.get_file_extension(subject)
            if "nii" not in extension:
                imed_inference.pred_to_png(pred_list,
                                           target_list,
                                           str(Path(pred_path, subject)).replace(extension, ''),
                                           suffix="_pred.png")


def run_command(context: dict, n_gif=0, thr_increment=None, resume_training=False, no_patch=False, overlap_2d=None):
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
        resume_training (bool): Load a saved model ("checkpoint.pth.tar" in the output directory specified with flag
            "--path-output" or via the config file "output_path". This training state is saved everytime a new best
            model is saved in the log argument) for resume training directory.
        no_patch (bool): If True, 2D patches are not used while segmenting with models trained with patches
            (command "--segment" only). Default: False (i.e. segment with patches). The "no_patch" option supersedes
            the "overlap_2D" option.
        overlap_2d (list of int): Custom overlap for 2D patches while segmenting (command "--segment" only).
            Default model overlap is used otherwise.

    Returns:
        float or pandas.DataFrame or None:
            * If "train" command: Returns floats: best loss score for both training and validation.
            * If "test" command: Returns a pandas Dataframe: of metrics computed for each subject of
              the testing sub-dataset and return the prediction metrics before evaluation.
            * If "segment" command: No return value.

    """
    # Command can only be: segment/train/test
    command: str = copy.deepcopy(context[ConfigKW.COMMAND])

    #################################
    #  Loader Version Determination:
    #################################
    # If the new keywords are present, we use the new loader pathway. Ignoring all "contrast_params" keys etc for now.
    if context.get(DataloaderKW.DATASET_GROUPS) and context.get(DataloaderKW.EXPECTED_GT) and context.get(DataloaderKW.EXPECTED_INPUT):
        loader_version: str = LoaderParamsKW.MULTI_PATH_LOADER
    else:
        loader_version: str = LoaderParamsKW.TRADITIONAL_BIDS_LOADER
        logger.info(f"Using {loader_version} loader version because new V2 loader keywords are not present in the context.")

    #################################

    path_output = set_output_path(context)
    path_log: Path = Path(
        context.get('path_output'),
        context.get('log_file')
    )
    logger.remove()
    logger.add(str(path_log))
    logger.add(sys.stdout)

    # Create a log with the version of the Ivadomed software and the version of the Annexed dataset (if present)
    # create_dataset_and_ivadomed_version_log(context)

    cuda_available, device = imed_utils.define_device(context[ConfigKW.GPU_IDS][0])

    # BACKWARDS COMPATIBILITY: If bids_path is string, assign to list - Do this here so it propagates to all functions
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA] = imed_utils.format_path_data(
            context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA]
        )

    # Set Loader params leveraging information from Loader_Parameters for the BIDS Loader version.
    # Not needed in the alternative loader version.
    loader_params: dict = update_loader_params(context, command == "train", loader_version)

    # Get transforms dictionaries for each subdataset
    transform_train_params, \
    transform_valid_params, \
    transform_test_params = imed_transforms.get_subdatasets_transforms(context[ConfigKW.TRANSFORMATION])

    # Update and populate model parameters AND loader parameters
    model_params, loader_params = set_model_params(context, loader_params)

    if command == CommandKW.SEGMENT and loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:
        run_segment_command(context, model_params, no_patch, overlap_2d)
        return

    if loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:

        # BIDSDataframe of all image files
        # Indexing of derivatives is True for commands train and test
        # split_method is used for removing unused subject files in bids_df for commands train and test
        bids_df = BidsDataframe(
            loader_params,
            path_output,
            derivatives=True,
            split_method=context.get(ConfigKW.SPLIT_DATASET).get(SplitDatasetKW.SPLIT_METHOD)
        )

        # Get subject filenames lists. "segment" command uses all participants of data path, hence no need to split
        train_lst, valid_lst, test_lst = imed_loader_utils.get_subdatasets_subject_files_list(context[ConfigKW.SPLIT_DATASET],
                                                                                              bids_df.df,
                                                                                              path_output,
                                                                                              context.get(ConfigKW.LOADER_PARAMETERS).get(
                                                                                                  LoaderParamsKW.SUBJECT_SELECTION))

        # Generating sha256 for the training files into the bids DataFrame
        imed_utils.generate_sha_256(context, bids_df.df, train_lst)

    testing_params, transformation_dict = generate_test_params_dict(
        context,
        loader_params,
        transform_test_params,
        transform_train_params
    )

    # Display transforms for the various subsets
    if command == CommandKW.TRAIN:
        imed_utils.display_selected_transfoms(transform_train_params, dataset_type=[DatasetTypeKW.TRAINING])
        imed_utils.display_selected_transfoms(transform_valid_params, dataset_type=[DatasetTypeKW.VALIDATION])
    elif command == CommandKW.TEST:
        imed_utils.display_selected_transfoms(transformation_dict, dataset_type=[DatasetTypeKW.TESTING])

    # Check if multiple raters
    check_multiple_raters(command==CommandKW.TRAIN, loader_params)

    # Training when only with BIDS and via the older path.
    if command == CommandKW.TRAIN:
        if loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:
            best_training_dice, \
            best_training_loss, \
            best_validation_dice, \
            best_validation_loss, \
            ds_valid = execute_training_bids(
                    bids_df,
                    context,
                    loader_params,
                    model_params,
                    n_gif,
                    path_output,
                    resume_training,
                    train_lst,
                    transform_train_params,
                    transform_valid_params,
                    valid_lst
                )
        elif loader_version == LoaderParamsKW.MULTI_PATH_LOADER:
            best_training_dice, \
            best_training_loss, \
            best_validation_dice, \
            best_validation_loss, \
            ds_valid, \
            model_params = \
                execute_multi_path_training(
                    context,
                    model_params,
                    n_gif,
                    path_output,
                    resume_training,
                    transform_train_params,
                    transform_valid_params,
                )
        else:
            raise ValueError("The loader version is not supported.")

    # Threshold Increment Analyses when dealing with BIDS dataset ONLY
    if thr_increment:
        if loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:
            execute_threshold_analyses(
                bids_df,
                command,
                context,
                cuda_available,
                device,
                ds_valid,
                loader_params,
                model_params,
                path_output,
                testing_params,
                thr_increment,
                train_lst,
                transform_valid_params,
                valid_lst
            )
        # todo: implement threshold increment analyses for multi-path loader

        else:
            raise ValueError("The threshold increment analyses is only supported for BIDS datasets.")



    # Early return training results when on the bids loading path.
    if command == CommandKW.TRAIN:
        return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss

    if command == CommandKW.TEST:
        if loader_version == LoaderParamsKW.TRADITIONAL_BIDS_LOADER:
            df_results, pred_metrics = execute_traditional_test(bids_df,
                                                                context,
                                                                cuda_available,
                                                                device,
                                                                loader_params,
                                                                model_params,
                                                                path_output,
                                                                test_lst,
                                                                testing_params,
                                                                transformation_dict)
            return df_results, pred_metrics

        # todo: implement test for multi-path loader

        else:
            raise ValueError(f"The loader version {loader_version} is not supported for testing currently.")


def generate_test_params_dict(context, loader_params, transform_test_params, transform_train_params):
    # TESTING PARAMS
    # Aleatoric uncertainty
    if context[ConfigKW.UNCERTAINTY][UncertaintyKW.ALEATORIC] \
            and context[ConfigKW.UNCERTAINTY][UncertaintyKW.N_IT] > 0:
        transformation_dict = transform_train_params
    else:
        transformation_dict = transform_test_params
    # UndoTransform Function to undo transforms
    undo_transforms = imed_transforms.UndoCompose(imed_transforms.Compose(transformation_dict, requires_undo=True))
    # Testing related parameters
    # Used in Threshold analyses and Testing process
    testing_params: dict = copy.deepcopy(context[ConfigKW.TRAINING_PARAMETERS])
    testing_params.update({ConfigKW.UNCERTAINTY: context[ConfigKW.UNCERTAINTY]})
    testing_params.update({
        LoaderParamsKW.TARGET_SUFFIX: loader_params[LoaderParamsKW.TARGET_SUFFIX],
        ConfigKW.UNDO_TRANSFORMS: undo_transforms,
        LoaderParamsKW.SLICE_AXIS: loader_params[LoaderParamsKW.SLICE_AXIS]
    })
    return testing_params, transformation_dict


def execute_training_bids(bids_df, context, loader_params, model_params, n_gif, path_output,
                          resume_training, train_lst, transform_train_params, transform_valid_params, valid_lst):

    cuda_available, device = imed_utils.define_device(context[ConfigKW.GPU_IDS][0])

    # Get Validation BIDS dataset
    ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device,
                           'validation')
    # Get Training BIDS dataset
    ds_train = get_dataset(bids_df, loader_params, train_lst, transform_train_params, cuda_available, device,
                           'training')
    metric_fns = imed_metrics.get_metric_fns(ds_train.task)
    # FiLM not supported, not gonna normalize data
    train_onehotencoder = None
    # Model directory
    create_path_model(context, model_params, ds_train, path_output, train_onehotencoder)
    save_config_file(context, path_output)
    ###########################
    # Run the training and validation to gather the metrics
    # The main training process
    ###########################
    best_training_dice, \
    best_training_loss, \
    best_validation_dice, \
    best_validation_loss = imed_training.train(
        model_params=model_params,
        dataset_train=ds_train,
        dataset_val=ds_valid,
        training_params=context[ConfigKW.TRAINING_PARAMETERS],
        wandb_params=context.get(ConfigKW.WANDB),
        path_output=path_output,
        device=device,
        cuda_available=cuda_available,
        metric_fns=metric_fns,
        n_gif=n_gif,
        resume_training=resume_training,
        debugging=context[ConfigKW.DEBUGGING]
    )
    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss, ds_valid


def execute_multi_path_training(context: dict,
                                model_params: dict,
                                n_gif: int,
                                path_output: str,
                                resume_training: bool,
                                transform_train_params: dict,
                                transform_valid_params: dict):

    cuda_available, device = imed_utils.define_device(context.get(ConfigKW.GPU_IDS)[0])

    from ivadomed.loader.all_dataset_group import AllDatasetGroups

    # Build simple dict to contain key dictionary.
    all_data_groups_dict = {}
    all_data_groups_dict[DataloaderKW.DATASET_GROUPS] = context.get(DataloaderKW.DATASET_GROUPS)
    all_data_groups_dict[DataloaderKW.EXPECTED_INPUT] = context.get(DataloaderKW.EXPECTED_INPUT)
    all_data_groups_dict[DataloaderKW.EXPECTED_GT] = context.get(DataloaderKW.EXPECTED_GT)

    # Build the generalized configuration object
    # Build all dataset group object (FileDataset, BIDSDataset, RegexDataset etc, all being taken care of by this constructor)
    all_data = AllDatasetGroups(
        all_data_groups_dict,
        GeneralizedLoaderConfiguration(model_params)
    )

    # Aggregate train across AllDatasetGroups
    ds_train = ConsolidatedDataset.consolidate_AllDatasetGroups_to_a_specific_filedataset_type(all_data, DataloaderKW.TRAINING)
    ds_train.load_filenames()

    # Aggregate validation dataset across AllDatasetGroups
    ds_valid = ConsolidatedDataset.consolidate_AllDatasetGroups_to_a_specific_filedataset_type(all_data, DataloaderKW.VALIDATION)
    ds_valid.load_filenames()

    metric_fns = imed_metrics.get_metric_fns(ds_train.task)
    # If FiLM, normalize data
    if ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS]):
        model_params, ds_train, ds_valid, train_onehotencoder = \
            film_normalize_data(context, model_params, ds_train, ds_valid, path_output)
    else:
        train_onehotencoder = None
    # Model directory
    create_path_model(context, model_params, ds_train, path_output, train_onehotencoder)
    save_config_file(context, path_output)
    ###########################
    # Run the training and validation to gather the metrics
    # The main training process
    ###########################
    best_training_dice, \
    best_training_loss, \
    best_validation_dice, \
    best_validation_loss = imed_training.train(
        model_params=model_params,
        dataset_train=ds_train,
        dataset_val=ds_valid,
        training_params=context[ConfigKW.TRAINING_PARAMETERS],
        wandb_params=context.get(ConfigKW.WANDB),
        path_output=path_output,
        device=device,
        cuda_available=cuda_available,
        metric_fns=metric_fns,
        n_gif=n_gif,
        resume_training=resume_training,
        debugging=context[ConfigKW.DEBUGGING]
    )
    return best_training_dice, best_training_loss, best_validation_dice, best_validation_loss, ds_valid, model_params


def execute_threshold_analyses(bids_df, command, context, cuda_available, device, ds_valid, loader_params, model_params,
                               path_output, testing_params, thr_increment, train_lst, transform_valid_params,
                               valid_lst):
    # LOAD DATASET
    if command != 'train':  # If command == train, then ds_valid already load
        # Get Validation dataset
        ds_valid = get_dataset(bids_df, loader_params, valid_lst, transform_valid_params, cuda_available, device,
                               'validation')
    # Get Training dataset with no Data Augmentation
    ds_train = get_dataset(bids_df, loader_params, train_lst, transform_valid_params, cuda_available, device,
                           'training')
    # Choice of optimisation metric
    if model_params[ModelParamsKW.NAME] in imed_utils.CLASSIFIER_LIST:
        metric = MetricsKW.RECALL_SPECIFICITY
    else:
        metric = MetricsKW.DICE
    # Model path
    model_path = Path(path_output, "best_model.pt")
    # Run analysis
    thr = imed_testing.threshold_analysis(model_path=str(model_path),
                                          ds_lst=[ds_train, ds_valid],
                                          model_params=model_params,
                                          testing_params=testing_params,
                                          metric=metric,
                                          increment=thr_increment,
                                          fname_out=str(Path(path_output, "roc.png")),
                                          cuda_available=cuda_available)
    # Update threshold in config file
    context[ConfigKW.POSTPROCESSING][PostprocessingKW.BINARIZE_PREDICTION] = {BinarizeProdictionKW.THR: thr}
    save_config_file(context, path_output)


def execute_traditional_test(bids_df, context, cuda_available, device, loader_params, model_params, path_output,
                             test_lst, testing_params, transformation_dict):
    # LOAD DATASET
    # Warn user that the input-level dropout is set during inference
    if loader_params[LoaderParamsKW.IS_INPUT_DROPOUT]:
        logger.warning("Input-level dropout is set during testing. To turn this option off, set 'is_input_dropout'"
                       "to 'false' in the configuration file.")
    ds_test = imed_loader.load_dataset(bids_df, **{**loader_params, **{'data_list': test_lst,
                                                                       'transforms_params': transformation_dict,
                                                                       'dataset_type': 'testing',
                                                                       'requires_undo': True}}, device=device,
                                       cuda_available=cuda_available)
    eval_params = context[ConfigKW.EVALUATION_PARAMETERS]
    metric_fns = imed_metrics.get_metric_fns(ds_test.task, eval_params)

    if ModelParamsKW.FILM_LAYERS in model_params and any(model_params[ModelParamsKW.FILM_LAYERS]):
        ds_test, model_params = update_film_model_params(context, ds_test, model_params, path_output)
    # RUN INFERENCE
    pred_metrics = imed_testing.test(model_params=model_params,
                                     dataset_test=ds_test,
                                     testing_params=testing_params,
                                     path_output=path_output,
                                     device=device,
                                     cuda_available=cuda_available,
                                     metric_fns=metric_fns,
                                     postprocessing=context[ConfigKW.POSTPROCESSING])
    # RUN EVALUATION
    df_results = imed_evaluation.evaluate(bids_df, path_output=path_output,
                                          target_suffix=loader_params[LoaderParamsKW.TARGET_SUFFIX],
                                              eval_params=eval_params)
    return df_results, pred_metrics


def create_dataset_and_ivadomed_version_log(context: dict):
    path_data = context.get(ConfigKW.LOADER_PARAMETERS).get(LoaderParamsKW.PATH_DATA)

    ivadomed_version = imed_utils._version_string()
    datasets_version = []

    if isinstance(path_data, str):
        datasets_version = [imed_utils.__get_commit(path_to_git_folder=path_data)]
    elif isinstance(path_data, list):
        for Dataset in path_data:
            datasets_version.append(imed_utils.__get_commit(path_to_git_folder=Dataset))

    path_log = Path(context.get(ConfigKW.PATH_OUTPUT), 'version_info.log')

    try:
        f = path_log.open(mode="w")
    except OSError as err:
        logger.error(f"OS error: {err}")
        raise Exception("Have you selected a log folder, and do you have write permissions for that folder?")

    # IVADOMED
    f.write('IVADOMED TOOLBOX\n----------------\n(' + ivadomed_version + ')')

    # DATASETS
    path_data = imed_utils.format_path_data(path_data)
    f.write('\n\n\nDATASET VERSION\n---------------\n')

    f.write('The following BIDS dataset(s) were used for training.\n')

    for i_dataset in range(len(path_data)):
        if datasets_version[i_dataset] not in ['', '?!?']:
            f.write(str(i_dataset + 1) + '. ' + path_data[i_dataset] + ' - Dataset Annex version: ' + datasets_version[
                i_dataset] + '\n')
        else:
            f.write(str(i_dataset + 1) + '. ' + path_data[i_dataset] + ' - Dataset is not Annexed.\n')

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

    context[ConfigKW.COMMAND] = imed_utils.get_command(args, context)
    context[ConfigKW.PATH_OUTPUT] = imed_utils.get_path_output(args, context)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA] = imed_utils.get_path_data(args, context)

    # Run command
    run_command(context=context,
                n_gif=args.gif if args.gif is not None else 0,
                thr_increment=args.thr_increment if args.thr_increment else None,
                resume_training=bool(args.resume_training),
                no_patch=bool(args.no_patch),
                overlap_2d=args.overlap_2d if args.overlap_2d else None)


if __name__ == "__main__":
    run_main()
