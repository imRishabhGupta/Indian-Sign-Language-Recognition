import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sklearn.metrics as sm

sess = tf.InteractiveSession()

train = pd.read_csv("train60.csv")
test  = pd.read_csv("train40.csv")

train_images = train.iloc[:,1:].values
train_images = train_images.astype(np.float)
train_images = np.multiply(train_images, 1.0 / 255.0)
labels_flat = train[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

test_images = test.iloc[:,1:].values
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)
label_test = test[[0]].values.ravel()
#labels_flat2 = test[[0]].values.ravel()
#labels_count2 = np.unique(labels_flat2).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

train_labels = dense_to_one_hot(labels_flat, labels_count)
train_labels = train_labels.astype(np.uint8)

#test_labels = dense_to_one_hot(labels_flat2, labels_count2)
#test_labels = test_labels.astype(np.uint8)

#validation_images = images[:VALIDATION_SIZE]
#validation_labels = labels[:VALIDATION_SIZE]

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

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
y_ = tf.placeholder(tf.float32, shape=[None, 24])

W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])

x_image = tf.reshape(x, [-1,96,96,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 16, 32])
b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 32, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([6 * 6 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool4, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 24])
b_fc2 = bias_variable([24])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

tf.add_to_collection('vars', W_conv1)
tf.add_to_collection('vars', W_conv2)
tf.add_to_collection('vars', W_conv3)
tf.add_to_collection('vars', W_conv4)
tf.add_to_collection('vars', b_conv1)
tf.add_to_collection('vars', b_conv2)
tf.add_to_collection('vars', b_conv3)
tf.add_to_collection('vars', b_conv4)
tf.add_to_collection('vars', W_fc1)
tf.add_to_collection('vars', b_fc1)
tf.add_to_collection('vars', W_fc2)
tf.add_to_collection('vars', b_fc2)
saver = tf.train.Saver()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predict = tf.argmax(y_conv,1)

sess.run(tf.global_variables_initializer())

print("starts")

for i in range(20000):
  batch_xs, batch_ys = next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

BATCH_SIZE = 50
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]):
    predicted_lables[i : (i+1)] = predict.eval(feed_dict={x: test_images[i : (i+1)], keep_prob: 1.0})
      
np.savetxt('submission_cnn.csv', np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print(sm.accuracy_score(label_test,predicted_lables))
print(sm.precision_score(label_test,predicted_lables,average='micro'))

saver.save(sess, os.path.join(os.getcwd(), 'trained_variables2.ckpt'))
