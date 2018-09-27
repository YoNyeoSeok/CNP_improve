import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_model",
        help='load model')
args = parser.parse_args()

model = torch.load(args.load_model)
for param in model.parameters():
    print(param)
