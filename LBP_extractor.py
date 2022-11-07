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
    lbp = local_binary_pattern(image, numPoints, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(), density=True,
                             bins=range(0, numPoints + p_1),
                             range=range(0, numPoints + p_2))
    # normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def createLBPData(path, output, bufSize, method):
    LBP_extractors = [(8, 1.0), (16, 2.0), (24, 3.0)]
    labels = [p.split(os.path.sep)[-2] for p in path]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    for numPoints, radius in tqdm(LBP_extractors):
        lbp_points = numPoints + numPoints * (numPoints - 2) + 3
        print(lbp_points)
        if method == 'uniform':
            lbp_points = numPoints + 2
        dataset = HDF5DatasetWriter((len(path), lbp_points),
                                    output + f"/LBP_{method}_{numPoints}",
                                    dataKey=f"features",
                                    buffSize=bufSize)
        dataset.storeClassLabels(le.classes_)
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
