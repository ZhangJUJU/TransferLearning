import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE

from load_dataset import *
BATCH_SIZE=4000
learningrate=0.00001
lamda=10
miu=10
MAX_STEP=500
Y = tf.placeholder('float', shape=[None, 10])
X = tf.placeholder('float', shape=[None, 256])
source_image,source_label=read_date('usps')#usps
target_image_all, target_label_all = read_date('mnist-fake')#mnist-fake
target_image = target_image_all[:55000, :]
target_label = target_label_all[:55000, :]

target_image_all_real, target_label_all_real = read_date('mnist-real')#mnist-real
target_image_test = target_image_all_real[55000:, :]
target_label_test = target_label_all_real[55000:, :]
def createMmetrix():
    mat1 = tf.constant(np.float32(1) / (np.square(BATCH_SIZE / 2)), shape=[BATCH_SIZE / 2, BATCH_SIZE / 2],\
                       dtype=tf.float32)
    mat2=-mat1
    mat3=tf.concat([mat1,mat2],1)
    mat4=tf.concat([mat2,mat1],1)
    mat5=tf.concat([mat3,mat4],0)
    return mat5
M=createMmetrix()
sess = tf.Session()
print(sess.run(M))
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# first convolutinal layer
w_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(X, [-1, 16, 16, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([4 * 4 * 64, 256])
b_fc1 = bias_variable([256])
h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# softmax layer
w_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)
label = tf.argmax(y_conv, 1)
obj_func = -tf.reduce_sum(Y * tf.log(y_conv)) + tf.constant(lamda, dtype=tf.float32) * tf.trace(
    tf.matmul(tf.matmul(h_fc1, M, transpose_a=True), h_fc1)) + tf.constant(miu, dtype=tf.float32) * tf.trace(
    tf.matmul(tf.matmul(y_conv, M, transpose_a=True), y_conv))
M_MMD=tf.constant(lamda, dtype=tf.float32) * tf.trace(
    tf.matmul(tf.matmul(h_fc1, M, transpose_a=True), h_fc1))
C_MMD=tf.constant(miu, dtype=tf.float32) * tf.trace(
    tf.matmul(tf.matmul(y_conv, M, transpose_a=True), y_conv))
optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(obj_func)
get_feature=h_fc1
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for echo in np.arange(20):
    for step in np.arange(MAX_STEP):
        source_image_batch, source_label_batch = get_minibatch(source_image, source_label, BATCH_SIZE)
        target_image_batch, target_label_batch = get_minibatch(target_image, target_label, BATCH_SIZE)
        total_image_batch=np.vstack([source_image_batch,target_image_batch])
        total_label_batch=np.vstack([source_label_batch,target_label_batch])
        _, train_acc,train_loss,m,c = sess.run([optimizer,accuracy,obj_func,M_MMD,C_MMD],
                                            feed_dict={X: total_image_batch,Y: total_label_batch})
        if(step%10 == 0):
            print('step=%d,...loss=%f,acc_train=%0.2f%%,MMMD=%f, CMMD=%f'%(step,train_loss,train_acc*100, m,c))

    new_label=sess.run([label],feed_dict={X: target_image_all})
    new_label = dense_to_one_hot(np.transpose(np.array(new_label)),10)
    target_label=new_label[:55000, :]
    target_acc = sess.run([accuracy],feed_dict={X: target_image_test,Y:target_label_test})
    print('DA ACC=%',target_acc)

