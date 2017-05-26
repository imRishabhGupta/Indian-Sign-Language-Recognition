import pandas as pd
from numpy._distributor_init import NUMPY_MKL
from sklearn import svm
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
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

def calc_accuracy(method,label_test,pred):
	print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
	print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
	print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
	print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))


def run_svm():
	clf=svm.SVC(decision_function_shape='ovo')
	print("svm started")
	clf.fit(x,y)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_svm.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("SVM",label_test,pred)

def run_lr():
	clf = lr()
	print("lr started")
	clf.fit(x,y)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_lr.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("Logistic regression",label_test,pred)


def run_nb():
	clf = nb()
	print("nb started")
	clf.fit(x,y)
	#print(clf.classes_)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_nb.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("Naive Bayes",label_test,pred)


def run_knn():
	clf=knn(n_neighbors=3)
	print("knn started")
	clf.fit(x,y)
	#print(clf.classes_)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_knn.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("K nearest neighbours",label_test,pred)

run_svm()
run_knn()
run_nb()
run_lr()
