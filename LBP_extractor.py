import os
from hdf5DatasetWriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
import numpy as np
import cv2
from tqdm.auto import tqdm
np.random.seed(42)


def extract(image, numPoints, radius, method, eps=1e-7):
    if method == "uniform":
        p_1, p_2 = 3, 2
    else:
        p_1, p_2 = numPoints * (numPoints - 2) + 4, numPoints * (numPoints - 2) + 3
    # EXTRACT LBP FEATURES BASED ON THE IMAGE
    lbp = local_binary_pattern(image, numPoints, radius, method=method)
    # CALCULATE HISTOGRAM FROM THE LBP
    (hist, _) = np.histogram(lbp.ravel(), density=True,
                             bins=range(0, numPoints + p_1),
                             range=range(0, numPoints + p_2))
    # normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def createLBPData(path, output, bufSize, method):
    # DIFFERENT LBP VARIATIONS
    LBP_extractors = [(8, 1.0), (16, 2.0), (24, 3.0)]
    # GET THE IMAGES' LABELS
    labels = [p.split(os.path.sep)[-2] for p in path]
    # CREATE LABEL ENCODER
    le = LabelEncoder()
    # CONVERT LABELS' NAME INTO NUMERIC FORMAT
    labels = le.fit_transform(labels)
    # PERFORM LBP FOR EACH VARIATION TO THE DATASET
    os.mkdir(output + f'/{method}')
    for numPoints, radius in tqdm(LBP_extractors):
        lbp_points = numPoints + numPoints * (numPoints - 2) + 3
        print(lbp_points)
        if method == 'uniform':
            lbp_points = numPoints + 2
        # CREATE HDF5 FILE TO STORE THE LBP FEATURES
        dataset = HDF5DatasetWriter((len(path), lbp_points),
                                    output + f'/{method}' + f"/LBP_{method}_{numPoints}",
                                    dataKey=f"features",
                                    buffSize=bufSize)
        # STORE THE LABELS' NAMES INTO HDF5
        dataset.storeClassLabels(le.classes_)
        # PROCESSING IMAGES BY BATCH
        for i in tqdm(range(0, len(path), bufSize), leave=False):
            batch_images = path[i: i + bufSize]
            batch_labels = labels[i: i + bufSize]
            processed_batch = []
            for _, imgPath in tqdm(enumerate(batch_images), leave=False):
                img = cv2.imread(imgPath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = extract(img.copy(), numPoints, radius, method)
                processed_batch.append(res)
            processed_batch = np.vstack(processed_batch)
            dataset.add(processed_batch, batch_labels)
        dataset.close()
