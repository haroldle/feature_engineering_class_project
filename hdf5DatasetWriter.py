import h5py
import os

# taken from a book that Le buy
# @book{rosebrock_dl4cv,
#   author={Rosebrock, Adrian},
#   title={Deep Learning for Computer Vision with Python},
#   year={2019},
#   edition={3.0.0},
#   publisher={PyImageSearch.com}
# }
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("Cannot overwritten exist path", outputPath)
        # Setting the storage
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        self.bufSize = buffSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    # setting the add function to add data to the storage
    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    # if the temporary storage meets the total number of items => write them to disk
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx: i] = self.buffer["data"]
        self.labels[self.idx: i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    # create function that store the class label into the storage
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
