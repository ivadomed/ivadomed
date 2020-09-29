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


def convert_pytorch_to_onnx(model, dimension, gpu=0):
    """Convert PyTorch model to ONNX.

    The integration of Deep Learning models into the clinical routine requires cpu optimized models. To export the
    PyTorch models to `ONNX <https://github.com/onnx/onnx>`_ format and to run the inference using
    `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_ is a time and memory efficient way to answer this need.

    This function converts a model from PyTorch to ONNX format, with information of whether it is a 2D or 3D model
    (``-d``).

    Args:
        model (string): Model filename. Flag: ``--model``, ``-m``.
        dimension (int): Indicates whether the model is 2D or 3D. Choice between 2 or 3. Flag: ``--dimension``, ``-d``
        gpu (string): GPU ID, if available. Flag: ``--gpu``, ``-g``
    """
    if torch.cuda.is_available():
        device = "cuda:" + str(gpu)
    else:
        device = "cpu"

    model_net = torch.load(model, map_location=device)
    dummy_input = torch.randn(1, 1, 96, 96, device=device) if dimension == 2 \
                  else torch.randn(1, 1, 96, 96, 96, device=device)
    imed_utils.save_onnx_model(model_net, dummy_input, model.replace("pt", "onnx"))


def main():
    imed_utils.init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()
    fname_model = args.model
    dimension = int(args.dimension)
    gpu = str(args.gpu)
    # Run Script
    convert_pytorch_to_onnx(fname_model, dimension, gpu)


if __name__ == '__main__':
    main()
