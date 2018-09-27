import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)

args = parser.parse_args()
model = torch.load(args.model)

for param in model.parameters():
    print(param[0])
