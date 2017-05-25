import pandas as pd
from numpy._distributor_init import NUMPY_MKL
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
import sklearn.metrics as sm

train = pd.read_csv("train60.csv")
print(train.head())
y = np.array(train.pop('label'))
x = np.array(train)/255.
print("here1")
#train = dataset.iloc[:,1:].values
test = pd.read_csv("train40.csv")
label_test=np.array(test.pop('label'))
x_ = np.array(test)/255.
print("here2")
clf=knn(n_neighbors=3)
print("here3")
clf.fit(x,y)
print(clf.classes_)
#print clf.n_layers_
pred=clf.predict(x_)
print(pred)
np.savetxt('submission_knn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print (sm.accuracy_score(label_test,pred))
print (sm.precision_score(label_test,pred,average='micro'))
