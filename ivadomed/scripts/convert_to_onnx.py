import argparse
import torch

from ivadomed import utils as imed_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, type=str, help="Path to .pt model.")
    parser.add_argument("-d", "--dimension", dest="dimension", required=True,
                        type=int, help="Input dimension (2 for 2D inputs, 3 for 3D inputs).")
    parser.add_argument("-g", "--gpu", dest="gpu", default=0, type=str, help="GPU number if available.")
    return parser


def convert_pytorch_to_onnx(fname_model, dimension, gpu=0):
    """Convert PyTorch model to ONNX.

    The integration of Deep Learning models into the clinical routine requires cpu optimized models. To export the
    PyTorch models to `ONNX <https://github.com/onnx/onnx>`_ format and to run the inference using
    `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_ is a time and memory efficient way to answer this need.

    This function converts a model from PyTorch to ONNX format, with information of whether it is a 2D or 3D model
    (``-d``).

    Args:
        fname_model (string): Model filename.
        dimension (int): Indicates whether the model is 2D or 3D. Choice between 2 or 3.
        gpu (string): GPU ID, if available
    """
    if torch.cuda.is_available():
        device = "cuda:" + str(gpu)
    else:
        device = "cpu"

    model = torch.load(fname_model, map_location=device)
    dummy_input = torch.randn(1, 1, 96, 96, device=device) if dimension == 2 \
                  else torch.randn(1, 1, 96, 96, 96, device=device)
    imed_utils.save_onnx_model(model, dummy_input, fname_model.replace("pt", "onnx"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    fname_model = args.model
    dimension = int(args.dimension)
    gpu = str(args.gpu)
    # Run Script
    convert_pytorch_to_onnx(fname_model, dimension, gpu)
