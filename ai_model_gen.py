import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob
import os



def create_model(filename):
    return

def main():
    dir_path = 'models'

    pth_files = glob.glob(dir_path + "/*.pth")
    for pth_file in pth_files:
        os.remove(pth_file)
        print(f"Deleted {pth_file}")

    csv_list = glob.glob('datasets/*.csv')

    for csv_file in csv_list:
        create_model(csv_file)