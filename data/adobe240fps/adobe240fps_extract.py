import argparse
import os
import os.path
from shutil import rmtree, move
import random
import multiprocessing as mp
from multiprocessing import Pool

import cv2
from tqdm import tqdm

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--videos_folder", type=str, required=True, help='path to the folder containing videos')
parser.add_argument("--dataset_folder", type=str, required=True, help='path to the output dataset folder')
parser.add_argument("--img_width", type=int, default=640, help="output image width")
parser.add_argument("--img_height", type=int, default=360, help="output image height")
args = parser.parse_args()

def extract_file_frames(param):
    video_path, out_path, out_pattern = param
    print(video_path)

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=count) as pbar:
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (args.img_width, args.img_height))
            cv2.imwrite(
                os.path.join(out_path, out_pattern.format(i)),
                frame
            )

            pbar.update(1)

def extract_frames(videos, inDir, outDir):
    """
    Converts all the videos passed in `videos` list to images.

    Parameters
    ----------
        videos : list
            name of all video files.
        inDir : string
            path to input directory containing videos in `videos` list.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        None
    """

    params = [(os.path.join(inDir, video), os.path.join(outDir, os.path.splitext(video)[0]), '{:04d}.jpg') for video in videos]

    pool = Pool(int(mp.cpu_count() * 1.5)) # use some more processes for faster io
    pool.map(extract_file_frames, params)

def main():
    # Create dataset folder if it doesn't exist already.
    if not os.path.isdir(args.dataset_folder):
        os.mkdir(args.dataset_folder)

    trainPath        = os.path.join(args.dataset_folder, "train")
    testPath         = os.path.join(args.dataset_folder, "test")
    if not os.path.isdir(trainPath):
        os.mkdir(trainPath)
    if not os.path.isdir(testPath):
        os.mkdir(testPath)

    f = open("test_list.txt", "r")
    videos = f.read().split('\n')
    extract_frames(videos, args.videos_folder, testPath)

    f = open("train_list.txt", "r")
    videos = f.read().split('\n')
    extract_frames(videos, args.videos_folder, trainPath)

main()
