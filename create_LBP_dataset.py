import argparse
from imutils import paths
import random
import LBP_extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", help="path to input dataset")
    ap.add_argument("-o", "--output", dest="OUTPUT", help="path to output dataset")

    args = vars(ap.parse_args())

    imagePaths = list(paths.list_images(args["dataset"]))
    random.shuffle(imagePaths)
    LBP_extractor.createLBPData(imagePaths, args["OUTPUT"], 32, "uniform")
    LBP_extractor.createLBPData(imagePaths, args["OUTPUT"], 32, "nri_uniform")


if __name__ == "__main__":
    main()
