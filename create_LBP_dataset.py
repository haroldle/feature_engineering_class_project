import argparse
from imutils import paths
import random
import LBP_extractor


def main():
    # CREATE ARGUMENT PARSER TO GET THE DATA PATH AND OUTPUT PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", help="path to input dataset")
    ap.add_argument("-o", "--output", dest="OUTPUT", help="path to output dataset")

    args = vars(ap.parse_args())
    # GETTING IMAGE PATHS
    imagePaths = list(paths.list_images(args["dataset"]))
    # SHUFFLING THE IMAGES
    random.shuffle(imagePaths)
    # EXTRACTING LBP ON THE DATASET UNIFORMLY AND NRI UNIFORMLY
    LBP_extractor.createLBPData(imagePaths, args["OUTPUT"], 32, "uniform")
    LBP_extractor.createLBPData(imagePaths, args["OUTPUT"], 32, "nri_uniform")


if __name__ == "__main__":
    main()
