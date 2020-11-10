import logging
import os
import subprocess
import joblib

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}

# List of classification models (ie not segmentation output)
CLASSIFIER_LIST = ['resnet18', 'densenet121']

logger = logging.getLogger(__name__)


def get_task(model_name):
    return "classification" if model_name in CLASSIFIER_LIST else "segmentation"


def cuda(input_var, cuda_available=True, non_blocking=False):
    """Passes input_var to GPU.

    Args:
        input_var (Tensor): either a tensor or a list of tensors.
        cuda_available (bool): If False, then return identity
        non_blocking (bool):

    Returns:
        Tensor
    """
    if cuda_available:
        if isinstance(input_var, list):
            return [t.cuda(non_blocking=non_blocking) for t in input_var]
        else:
            return input_var.cuda(non_blocking=non_blocking)
    else:
        return input_var


def unstack_tensors(sample):
    """Unstack tensors.

    Args:
        sample (Tensor):

    Returns:
        list: list of Tensors.
    """
    list_tensor = []
    for i in range(sample.shape[1]):
        list_tensor.append(sample[:, i, ].unsqueeze(1))
    return list_tensor


def save_onnx_model(model, inputs, model_path):
    """Convert PyTorch model to ONNX model and save it as `model_path`.

    Args:
        model (nn.Module): PyTorch model.
        inputs (Tensor): Tensor, used to inform shape and axes.
        model_path (str): Output filename for the ONNX model.
    """
    model.eval()
    dynamic_axes = {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width', 4: 'depth'}
    if len(inputs.shape) == 4:
        del dynamic_axes[4]
    torch.onnx.export(model, inputs, model_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': dynamic_axes, 'output': dynamic_axes})


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        Bool, device: True if cuda is available.
    """
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(gpu_id)
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))
    return cuda_available, device


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): Keys are param names and values are param values.
    """
    print('\nSelected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            print('\t{}: {}'.format(k, params[k]))


def display_selected_transfoms(params, dataset_type):
    """Display in terminal the selected transforms for a given dataset.

    Args:
        params (dict):
        dataset_type (list): e.g. ['testing'] or ['training', 'validation']
    """
    print('\nSelected transformations for the {} dataset:'.format(dataset_type))
    for k in list(params.keys()):
        print('\t{}: {}'.format(k, params[k]))


def plot_transformed_sample(before, after, list_title=[], fname_out="", cmap="jet"):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before (ndarray): Sample before transform.
        after (ndarray): Sample after transform.
        list_title (list of str): Sub titles of before and after, resp.
        fname_out (str): Output filename where the plot is saved if provided.
        cmap (str): Matplotlib colour map.
    """
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    plt.interactive(False)
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', cmap=cmap)
    plt.title(list_title[0], fontsize=20)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', cmap=cmap)
    plt.title(list_title[1], fontsize=20)

    if fname_out:
        plt.savefig(fname_out)
    else:
        matplotlib.use('TkAgg')
        plt.show()


def _git_info(commit_env='IVADOMED_COMMIT', branch_env='IVADOMED_BRANCH'):
    """Get ivadomed version info from GIT.

    This functions retrieves the ivadomed version, commit, branch and installation type.

    Args:
        commit_env (str):
        branch_env (str):
    Returns:
        str, str, str, str: installation type, commit, branch, version.
    """
    ivadomed_commit = os.getenv(commit_env, "unknown")
    ivadomed_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__ivadomed_dir__, ".git")):
        ivadomed_commit = __get_commit() or ivadomed_commit
        ivadomed_branch = __get_branch() or ivadomed_branch

    if ivadomed_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    path_version = os.path.join(__ivadomed_dir__, 'ivadomed', 'version.txt')
    with open(path_version) as f:
        version_ivadomed = f.read().strip()

    return install_type, ivadomed_commit, ivadomed_branch, version_ivadomed


def check_exe(name):
    """Ensure that a program exists.

    Args:
        name (str): Name or path to program.
    Returns:
        str or None: path of the program or None
    """

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(name)
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, name)
            if is_exe(exe_file):
                return exe_file

    return None


def __get_commit(path_to_git_folder=None):
    """Get GIT ivadomed commit.

    Args:
        path_to_git_folder (str): Path to GIT folder.
    Returns:
        str: git commit ID, with trailing '*' if modified.
    """
    if path_to_git_folder is None:
        path_to_git_folder = __ivadomed_dir__
    else:
        path_to_git_folder = os.path.abspath(os.path.expanduser(path_to_git_folder))

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def __get_branch():
    """Get ivadomed branch.

    Args:

    Returns:
        str: ivadomed branch.
    """
    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__ivadomed_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def _version_string():
    install_type, ivadomed_commit, ivadomed_branch, version_ivadomed = _git_info()
    if install_type == "package":
        return version_ivadomed
    else:
        return "{install_type}-{ivadomed_branch}-{ivadomed_commit}".format(**locals())


__ivadomed_dir__ = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
__version__ = _version_string()


def init_ivadomed():
    """Initialize the ivadomed for typical terminal usage."""
    # Display ivadomed version
    logger.info('\nivadomed ({})\n'.format(__version__))
