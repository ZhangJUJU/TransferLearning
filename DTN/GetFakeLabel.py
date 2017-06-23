from load_dataset import *

BATCH_SIZE=200
MAX_STEP=1000

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

class Model(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder('float', shape=[None, 256])
        self.Y = tf.placeholder('float', shape=[None, 10])
        self.lr = tf.placeholder('float')
        x_image=tf.reshape(self.X, [-1, 16, 16, 1])
        with tf.variable_scope('feature_extractor'):
            # first convolutinal layer
            w_conv1 = weight_variable([3, 3, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # second convolutional layer
            w_conv2 = weight_variable([3, 3, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        with tf.variable_scope('label_predictor'):
            # densely connected layer
            w_fc1 = weight_variable([4 * 4 * 64, 500])
            b_fc1 = bias_variable([500])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
            self.feature=h_fc1
            # softmax layer
            w_fc2 = weight_variable([500, 10])
            b_fc2 = bias_variable([10])
            self.y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)
            self.label = tf.argmax(self.y_conv, 1)
graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    lr = tf.placeholder(tf.float32, [])
    obj_func_sourceonly = -tf.reduce_sum(model.Y * tf.log(model.y_conv))
    optimizer_sourceonly = tf.train.GradientDescentOptimizer(lr).minimize(obj_func_sourceonly)

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(model.y_conv, 1), tf.argmax(model.Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

source_image, source_label = read_date(1)
target_image_all, target_label_all = read_date(0)
def getFakeLabel():

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        target_image = target_image_all[:45000, :]
        target_label = target_label_all[:45000, :]
        target_image_test = target_image_all[45000:, :]
        target_label_test = target_label_all[45000:, :]

        source_image_batch_add, source_label_batch_add = get_minibatch(source_image, source_label, BATCH_SIZE)
        for step in np.arange(MAX_STEP):
            p = float(step) / MAX_STEP
            lrate = 0.001 / (1. + 10 * p) ** 0.75
            source_image_batch, source_label_batch = get_minibatch(source_image, source_label, BATCH_SIZE)
            total_image_batch = np.vstack([source_image_batch, source_image_batch_add])
            total_label_batch = np.vstack([source_label_batch, source_label_batch_add])
            _, train_acc, train_loss = sess.run([optimizer_sourceonly, accuracy, obj_func_sourceonly],
                                                feed_dict={model.X: total_image_batch, model.Y: total_label_batch,
                                                           lr: lrate})
            if (step % 10 == 0):
                print('step=%d,loss=%f,acc=%0.2f%%,,lr=%f' % (step, train_loss, train_acc * 100, lrate))
        fake_label = sess.run([model.label], feed_dict={model.X: target_image_all})
        source_acc, source_fc = sess.run([accuracy, model.feature],
                                         feed_dict={model.X: source_image, model.Y: source_label})
        target_acc, target_fc = sess.run([accuracy, model.feature],
                                         feed_dict={model.X: target_image_test, model.Y: target_label_test})
    return fake_label,source_acc,target_acc
if __name__ == '__main__':
    fakelabel,sourceacc,targetacc=getFakeLabel()
    print(sourceacc,targetacc)
    fakelabel=np.transpose(np.array(fakelabel))
    fakedata_total=np.concatenate([fakelabel,target_image_all],axis=1)
    print(fakedata_total.shape)
    np.save('mnist16-fake',fakedata_total)