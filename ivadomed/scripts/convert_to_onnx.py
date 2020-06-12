##############################################################
#
# This converts a .pt model to ONNX format
#
# Usage: python scripts/convert_to_onnx.py -m path/to/model.pt -d 3
#
##############################################################

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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:" + str(args.gpu)
    else:
        device = "cpu"

    model = torch.load(args.model, map_location=device)
    dummy_input = torch.randn(1, 1, 96, 96, device=device) if args.dimension == 2 \
                  else torch.randn(1, 1, 96, 96, 96, device=device)
    imed_utils.save_onnx_model(model, dummy_input, args.model.replace("pt", "onnx"))
