import numpy as np
import cv2
import os
import csv
import sklearn.metrics as sm
from surf import func
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import random
import warnings
import pickle

path="train"
label=0

img_descs=[]
y=[]

def perform_data_split(X, y, training_idxs, test_idxs, val_idxs):
    """
    Split X and y into train/test/val sets
    Parameters:
    -----------
    X : eg, use img_bow_hist
    y : corresponding labels for X
    training_idxs : list/array of integers used as indicies for training rows
    test_idxs : same
    val_idxs : same
    Returns:
    --------
    X_train, X_test, X_val, y_train, y_test, y_val
    """
    X_train = X[training_idxs]
    X_test = X[test_idxs]
    X_val = X[val_idxs]

    y_train = y[training_idxs]
    y_test = y[test_idxs]
    y_val = y[val_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val

def train_test_val_split_idxs(total_rows, percent_test, percent_val):
    """
    Get indexes for training, test, and validation rows, given a total number of rows.
    Assumes indexes are sequential integers starting at 0: eg [0,1,2,3,...N]
    Returns:
    --------
    training_idxs, test_idxs, val_idxs
        Both lists of integers
    """
    if percent_test + percent_val >= 1.0:
        raise ValueError('percent_test and percent_val must sum to less than 1.0')

    row_range = range(total_rows)

    no_test_rows = int(total_rows*(percent_test))
    test_idxs = np.random.choice(row_range, size=no_test_rows, replace=False)
    # remove test indexes
    row_range = [idx for idx in row_range if idx not in test_idxs]

    no_val_rows = int(total_rows*(percent_val))
    val_idxs = np.random.choice(row_range, size=no_val_rows, replace=False)
    # remove validation indexes
    training_idxs = [idx for idx in row_range if idx not in val_idxs]

    print('Train-test-val split: %i training rows, %i test rows, %i validation rows' % (len(training_idxs), len(test_idxs), len(val_idxs)))

    return training_idxs, test_idxs, val_idxs

def cluster_features(img_descs, training_idxs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters

    # # Generate the SIFT descriptor features
    # img_descs = gen_sift_features(labeled_img_paths)
    #
    # # Generate indexes of training rows
    # total_rows = len(img_descs)
    # training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, percent_test, percent_val)

    # Concatenate all descriptors in the training set together
    training_descs = [img_descs[i] for i in training_idxs]
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)
    np.savetxt("descriptors.csv", all_train_descriptors, delimiter=",")

    if all_train_descriptors.shape[1] != 64:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # Cluster descriptors to get codebook
    print ('Using clustering model %s...' % repr(cluster_model))
    print ('Clustering on training set to get codebook of %i words' % n_clusters)

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print ('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('done generating BoW histograms.')

    return X, cluster_model

def run_svm(X_train, X_test, y_train, y_test, scoring,
    c_vals=[1, 5, 10], gamma_vals=[0.1, 0.01, 0.0001, 0.00001]):

    param_grid = [
    #   {'C': c_vals, 'kernel': ['linear']},
      {'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},
     ]

    svc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=scoring)
    svc.fit(X_train, y_train)
    print ('train score (%s):'%scoring, svc.score(X_train, y_train))
    test_score = svc.score(X_test, y_test)
    print ('test score (%s):'%scoring, test_score)

    print (svc.best_estimator_)

    return svc, test_score

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    svc.fit(X_train,y_train)
    y_pred=svc.predict(X_test)
    print('Accuracy Score with svm:')
    print(sm.accuracy_score(y_test,y_pred))

#creating desc for each file with label
for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
            for file in files:
                actual_path=path+"\\\\"+dirname+"\\\\"+file
                print(actual_path)
                des=func(actual_path)
                img_descs.append(des)
                y.append(label)
        label=label+1



#
y=np.array(y)
training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(len(img_descs), 0.4, 0.0)

#
X, cluster_model = cluster_features(img_descs, training_idxs, MiniBatchKMeans(n_clusters=100))
np.savetxt("histogram.csv", X, delimiter=",")

X_train, X_test, X_val, y_train, y_test, y_val = perform_data_split(X, y, training_idxs, test_idxs, val_idxs)



predict_svm(X_train, X_test,y_train, y_test)
'''    
svc, test_score=run_svm(X_train, X_test,y_train, y_test,None)

print(svc)
print(test_score)
pred=svc.predict(X_test)
print(sm.accuracy_score(y_test,pred))

'''
