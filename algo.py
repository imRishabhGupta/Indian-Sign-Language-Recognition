import pandas as pd
from numpy._distributor_init import NUMPY_MKL
from sklearn import svm
import numpy as np

train = pd.read_csv("train96.csv")
print(train.head())
y = np.array(train.pop('label'))
x = np.array(train)/255.
print("here1")
#train = dataset.iloc[:,1:].values
test = pd.read_csv("test.csv")
x_ = np.array(test)/255.
print("here2")
clf=svm.SVC(decision_function_shape='ovo')
print("here3")
clf.fit(x,y)
print(clf.classes_)
#print clf.n_layers_
pred=clf.predict(x_)
print(pred)
np.savetxt('submission_svm.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

