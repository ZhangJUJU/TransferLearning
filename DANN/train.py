import os
import sys
sys.path.append("../")
from flip_gradient import *
import tensorflow as tf
import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *
import tarfile
tar_mnist = tarfile.open('mnist16-one-shuffled.tar.gz')
tar_usps = tarfile.open('usps16-one-shuffled.tar.gz')
tar_mnist.extract('mnist16-one-shuffled.npy',path='date/')
tar_mnist.close()
tar_usps.extract('usps16-one-shuffled.npy',path='date/')
tar_usps.close()



# FIXME:60000,7291  55000 10000 5000
np.random.seed(0)
tf.set_random_seed(0)

# progress date
mnist_data=np.load('date/mnist16-one-shuffled.npy')
usps_data= np.load('date/usps16-one-shuffled.npy')

source_train=usps_data[:5500,1:].reshape(5500,16,16,1)*256
source_train=source_train.astype(np.uint8).repeat(3,axis=3)
source_train_labels=dense_to_one_hot(usps_data[:5500, 0], 10)

source_test=usps_data[5500:6500,1:].reshape(1000,16,16,1)*256
source_test=source_test.astype(np.uint8).repeat(3,axis=3)
source_test_labels=dense_to_one_hot(usps_data[5500:6500, 0], 10)

target_train=mnist_data[:55000,1:].reshape(55000,16,16,1)*256
target_train=target_train.astype(np.uint8).repeat(3,axis=3)
target_train_labels=dense_to_one_hot(mnist_data[:55000,0],10)

target_test=mnist_data[50000:,1:].reshape(10000,16,16,1)*256
target_test=target_test.astype(np.uint8).repeat(3,axis=3)
target_test_labels=dense_to_one_hot(mnist_data[50000:,0],10)

# Compute pixel mean for normalizing data
pixel_mean = np.vstack([source_train, target_train]).mean(axis=(0, 1, 2))

# Create a mixed dataset for T-SNE visualization
num_test = 500
combined_test_imgs = np.vstack([source_test[:num_test], target_test[:num_test]])
combined_test_labels = np.vstack([source_test_labels[:num_test], source_test_labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                  np.tile([0., 1.], [num_test, 1])])

imshow_grid(source_train)
imshow_grid(target_train)

batch_size=128
class targetodel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 16, 16, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        #X_input=(tf.cast(self.X, tf.float32))
        X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 4 * 4 * 48])

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size / 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size / 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([4 * 4 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)
            #feat=tf.negative(self.feature) * self.l
            d_W_fc0 = weight_variable([4 * 4 * 48, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)


# Build the model graph

graph = tf.get_default_graph()
with graph.as_default():
    model = targetodel()
    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

# Hyperparams
batch_size = 256
num_steps = 10000

def train_and_eval(training_mode, graph, model, verbose=False):
    """ Helper to run the model with different training modes. """
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Batch generators
        gen_source_batch = batch_generator(
            [source_train, source_train_labels], batch_size / 2)
        gen_target_batch = batch_generator(
            [target_train, target_train_labels], batch_size / 2)
        gen_source_only_batch = batch_generator(
            [source_train, source_train_labels], batch_size,False)
        gen_target_only_batch = batch_generator(
            [target_train, target_train_labels], batch_size,False)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                   np.tile([0., 1.], [batch_size / 2, 1])])

        # Training loop
        for i in range(num_steps):
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p) ** 0.75

            # Training step
            if training_mode == 'dann':
                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc = \
                    sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                             feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                        model.train: True, model.l: l, learning_rate: lr})

                if verbose and i % 100 == 0:
                    print('loss: %f  d_acc: %f  p_acc: %f  progress->1: %f  lambda: %f  learning rate: %f' % \
                          (batch_loss, d_acc, p_acc, p, l, lr))

            elif training_mode == 'source':
                X, y = gen_source_only_batch.next()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

            elif training_mode == 'target':
                X, y = gen_target_only_batch.next()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})

        # Compute the final evaluation on test data
        source_acc = sess.run(label_acc,
                              feed_dict={model.X: source_test, model.y: source_test_labels,
                                         model.train: False})

        target_acc = sess.run(label_acc,
                              feed_dict={model.X: target_test, model.y: target_test_labels,
                                         model.train: False})

        test_domain_acc = sess.run(domain_acc,
                                   feed_dict={model.X: combined_test_imgs,
                                              model.domain: combined_test_domain, model.l: 1.0})

        test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})

    return source_acc, target_acc, test_domain_acc, test_emb


print '\nSource only training'
source_acc, target_acc, _, source_only_emb = train_and_eval('source', graph, model,verbose=True)
print 'Source (MNIST) accuracy:', source_acc
print 'Source only model test on Target (MNIST-M) accuracy:', target_acc

print '\nDomain adaptation training'
source_acc, target_acc, d_acc, dann_emb = train_and_eval('dann', graph, model,verbose=True)
print 'Source (MNIST) accuracy:', source_acc
print 'DANN model test on Target (MNIST-M) accuracy:', target_acc
print 'Domain accuracy:', d_acc


def plot_embedding(X, y, d, title=None):
    """ Plot an embedding X with the class label y colored by the domain d. """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
source_only_tsne = tsne.fit_transform(source_only_emb)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne = tsne.fit_transform(dann_emb)

plot_embedding(source_only_tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'Source only')
plot_embedding(dann_tsne, combined_test_labels.argmax(1),
               combined_test_domain.argmax(1), 'Domain Adaptation')