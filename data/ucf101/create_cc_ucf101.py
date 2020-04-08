import argparse

import os
import os.path
from shutil import copy

import cv2
from tqdm import tqdm

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_folder", type=str, required=True, help='path to the folder containing ucf101 pictures')
parser.add_argument("--save_folder", type=str, required=True, help='path to the output folder')
args = parser.parse_args()

def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    folder_names = os.listdir(args.root_folder)
    
    for folder_name in tqdm(folder_names):
        input_folder = os.path.join(args.root_folder, folder_name)
        output_folder = os.path.join(args.save_folder, folder_name)
        
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
            
        copy(os.path.join(input_folder, 'frame_00.png'), output_folder)
        copy(os.path.join(input_folder, 'frame_02.png'), output_folder)
        copy(os.path.join(input_folder, 'frame_01_gt.png'), os.path.join(output_folder, 'frame_01.png'))

main()
