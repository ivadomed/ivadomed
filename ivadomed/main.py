import sys
import json

import torch


def cmd_train(context):
    # Set the GPU
    gpu_number = int(context["gpu"])
    torch.cuda.set_device(gpu_number)

    print(context)
    return


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        return cmd_train(context)


if __name__ == "__main__":
    run_main()
