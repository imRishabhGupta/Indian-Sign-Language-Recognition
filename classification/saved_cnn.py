import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sklearn.metris as sm

test  = pd.read_csv("test.csv")
test_images = test.iloc[:,1:].values
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)
label_test = test[[0]].values.ravel()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
x = tf.placeholder(tf.float32, shape=[None, 9216])

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('trained_variables2.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_.shape)

W_conv1 = sess.run(all_vars[0])
b_conv1 = sess.run(all_vars[4])

x_image = tf.reshape(x, [-1,96,96,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = sess.run(all_vars[1])
b_conv2 = sess.run(all_vars[5])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = sess.run(all_vars[2])
b_conv3 = sess.run(all_vars[6])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = sess.run(all_vars[3])
b_conv4 = sess.run(all_vars[7])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = sess.run(all_vars[8])
b_fc1 = sess.run(all_vars[9])

h_pool2_flat = tf.reshape(h_pool4, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = sess.run(all_vars[10])
b_fc2 = sess.run(all_vars[11])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

predict = tf.argmax(y_conv,1)

predicted_lables = np.zeros(test_images.shape[0])
print("evaluating")
for i in range(0,test_images.shape[0]):
    predicted_lables[i : (i+1)] = predict.eval(feed_dict={x: test_images[i : (i+1)], 
                                                                                keep_prob: 1.0})
np.savetxt('submission_cnn111.csv', np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print(sm.accuracy_score(label_test,pred))
print(sm.precision_score(label_test,pred,average='micro'))