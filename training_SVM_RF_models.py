from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
import argparse
import os
import joblib


def grab_data(path, method, is_pca=False):
    db = []
    lb = None
    lb_n = None
    for i in path:
        # hdf5 file is just a dictionary with numpy where each key contain a numpy array like format.
        # like db['features'], db['label'], db['label_names']
        db_temp = h5py.File(i, 'r')
        # getting the features using the key name features;
        # features is a 2D array where columns represent the number of features and rows represent instances
        features = db_temp['features']

        # Doing PCA method on nri_uniform features if the number of nri_features > 50
        if method == "nri_uniform" and features.shape[1] > 59 and is_pca:
            pca = PCA(n_components=0.95, random_state=42)
            features = pca.fit_transform(features)

        db.append(features)
        # Grab labels and name of the labels
        if lb is None:
            lb = db_temp['labels']
            lb_n = db_temp['label_names']
    # db will contain multiple 2D arrays
    # features fusion: just stacking feature horizontally.
    # for example: a has shape (1090, 4) b has shape (1090, 5)
    # when we fuse a and b the result will be (1090, 9)
    db = np.hstack(db)

    # return data along with labels and the name of each label
    return db, np.array(lb), [str(lb_n[i]) for i in range(len(lb_n))]


def getFilesPaths(path):
    uniform = []
    nri_uniform = []

    for m, p in path:
        if m == 'uniform':
            uniform.append(p)
        else:
            nri_uniform.append(p)
    return uniform, nri_uniform


def train(db, labels, size, label_names, method, model_type):
    X_train, X_test, y_train, y_test = train_test_split(db, labels, test_size=1 - size, random_state=32)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=24)
    if model_type == 'SVM':
        params = {'C': [2 ** i for i in range(-2, 15)], 'degree': range(2, 15)}
        clf = SVC(kernel='poly')
    else:
        params = {'n_estimators': [i for i in range(500, 3000, 500)]}
        clf = RandomForestClassifier(class_weight='balanced')
    x = RandomizedSearchCV(estimator=clf,
                           param_distributions=params,
                           cv=rskf,
                           verbose=10,
                           scoring='balanced_accuracy',
                           n_jobs=6)
    x.fit(X_train, y_train)

    clf = x.best_estimator_
    report = classification_report(y_test, clf.predict(X_test), target_names=label_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'{model_type}_{method}.csv')
    joblib.dump(clf, f'{model_type}_{method}.sav')


def main():
    # run: python training_SVM_models.py -d LBP_processed_data -s 0.8
    # -d path to the preprocessed dataset
    # -s size of the training set

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path HDF5 dataset")
    ap.add_argument("-s", "--size", required=True, type=float, help="percentage of training dataset")
    ap.add_argument("-p", "--pca", type=int, help="Doing PCA on non rotation invariant dataset. 1 for yes, 0 for no")
    args = vars(ap.parse_args())

    # get the data paths
    data_paths = [(m, args['dataset'] + '/' + m + '/' + item)
                  for m in os.listdir(args['dataset'])
                  for item in os.listdir(args['dataset'] + '/' + m)]

    # get the path of each LBP feature
    uniform_ri, uniform_nri = getFilesPaths(data_paths)

    # initalize data for LBP uniform and LBP non-rotation invariant uniform
    db_uniform, labels, label_names = grab_data(uniform_ri, 'uniform', True if args['pca'] == 1 else False)
    db_nri, nri_labels, nri_label_names = grab_data(uniform_nri, 'nri_uniform', True if args['pca'] == 1 else False)

    # train Machine Learning model (SVM section)
    train(db_uniform, labels, args['size'], label_names, 'uniform', "SVM")
    train(db_nri, nri_labels, args['size'], nri_label_names, 'nri_uniform', "SVM")
    # train Machine Learning model (RF section)
    train(db_uniform, labels, args['size'], label_names, 'uniform', "RF")
    train(db_nri, nri_labels, args['size'], nri_label_names, 'nri_uniform', "RF")


if __name__ == '__main__':
    main()
