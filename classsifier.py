# Load pickled data
import pickle
import os.path as path
import os

# TODO: Fill this in based on where you saved the training and testing data
training_file = 'data/train.p'
validation_file = 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#####

import numpy as np

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
train_features = train['features']
train_labels = train['labels']
n_train = train_features.shape[0]

valid_features = valid['features']
valid_labels = valid['labels']
n_valid = valid_features.shape[0]

test_features = test['features']
test_labels = test['labels']

n_test = test_features.shape[0]

image_shape = train_features.shape[1:3]

n_classes = len(np.unique(train_labels))

print("Number of training examples =", n_train)
print("Number of valid examples =", n_valid)
print("Number of test examples =", n_test)

print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Chanel of image = ", train_features.shape[3])
print('Max feature in test = ', train_features.max())
print('Min feature in test = ', train_features.min())

####
import matplotlib.pyplot as plt
from skimage import transform
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split


# Visualizations will be shown in the notebook.


def hist_class(labels, ylabel, class_bin=n_classes):
    plt.subplot(2, 1, 1)
    plt.hist(labels, class_bin)
    plt.xlabel('class')
    plt.ylabel(ylabel)
    plt.show()


hist_class(train_labels, 'count in train')
hist_class(valid_labels, 'count in valid')
hist_class(test_labels, 'count in test')

####
combined_features = np.vstack((train_features, valid_features, test_features))
combined_labels = np.hstack((train_labels, valid_labels, test_labels))

combined_features, combined_labels = shuffle(combined_features, combined_labels)
hist = np.histogram(combined_labels, n_classes)
print(np.max(hist[0]))
print(np.min(hist[0]))

hist_class(combined_labels, 'count in combined')

####
# select 250 sample from every class
print(combined_labels.shape)
print((combined_labels == 1).nonzero()[0])
selected_position = []
for i in range(n_classes):
    selected_position.append((combined_labels == i).nonzero()[0][0:250])
print(np.array(selected_position).shape)

selected_position = np.array(selected_position).ravel()
balanced_features = combined_features[selected_position]
balanced_labels = combined_labels[selected_position]
print(balanced_features.shape)
print(balanced_labels.shape)

hist_class(balanced_labels, 'count in balanced')

####
import random

random.seed()

size = len(balanced_features)
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(balanced_features[random.randint(1, size)])
plt.savefig('randomImages.png')
plt.show()

####
import tensorflow as tf
import cv2
from skimage import exposure
from skimage import transform


def grayscale(imgs):
    result = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=gray.ndim)
        result.append(gray)
    return np.array(result)


def normalize(features):
    max = 255
    min = 0
    high_bound = 1
    low_bound = -1
    return low_bound + ((features - min) / (max - min)) * (high_bound - low_bound)


def equalize_hist(example):
    for i, j in enumerate(example):
        example[i] = exposure.equalize_hist(j)
    return example


####


def augment(examples, labels):
    # TODO rise a illegal argument exception,like java
    if examples.shape[0] != labels.shape[0]:
        return

    result_examples = []
    result_labels = []
    max_padding = 5
    for i, j in enumerate(examples):
        result_examples.append(j)

        result_examples.append(exposure.rescale_intensity(j))
        M = cv2.getRotationMatrix2D((16, 16), random.randint(-15, 15), 1)
        result_examples.append(cv2.warpAffine(j, M, (32, 32)))

        padded = cv2.copyMakeBorder(j, random.randint(0, max_padding),
                                    random.randint(0, max_padding),
                                    random.randint(0, max_padding),
                                    random.randint(0, max_padding),
                                    cv2.BORDER_REPLICATE)
        padded = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_LINEAR)
        result_examples.append(padded)
        result_labels.append(labels[i])
        result_labels.append(labels[i])
        result_labels.append(labels[i])
        result_labels.append(labels[i])

    return np.array(result_examples), np.array(result_labels)


balanced_features, balanced_labels = augment(balanced_features,
                                             balanced_labels)

size = len(balanced_features)

start = random.randint(1, size)
plt.subplot(1, 2, 1)
plt.imshow(balanced_features[start])
plt.subplot(1, 2, 2)
plt.imshow(balanced_features[start + 1])
plt.savefig('augmentedImage.jpg')
plt.show()

# processed_train_features = grayscale(train_features)
balanced_features = normalize(balanced_features)
assert balanced_features.shape[3] == 3

print('Precessed')

balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels)

b_train_valid_features, b_test_features, b_train_valid_labels, b_test_labels = train_test_split(balanced_features,
                                                                                                balanced_labels,
                                                                                                test_size=0.2)
b_train_features, b_valid_features, b_train_labels, b_valid_labels = train_test_split(b_train_valid_features,
                                                                                      b_train_valid_labels,
                                                                                      test_size=0.2)

####




####
X = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

keep_prob = tf.placeholder(dtype=tf.float32)
mu = 0
sigma = 0.1

# begin define model
# first layer conv,output 28x28x4
conv1_W = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mu, sigma))
conv1_b = tf.Variable(tf.truncated_normal([6], mu, sigma))

conv1 = tf.nn.conv2d(X,
                     conv1_W,
                     strides=[1, 1, 1, 1],
                     padding='VALID')
conv1 = tf.nn.bias_add(conv1, conv1_b)

conv1 = tf.nn.relu(conv1)
# first pool layer,output 14x14x4
pool1 = tf.nn.max_pool(conv1,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID')
# second conv layer,output 10x10x16
conv2_W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma))
conv2_b = tf.Variable(tf.truncated_normal([16], mu, sigma))
conv2 = tf.nn.conv2d(pool1,
                     conv2_W,
                     strides=[1, 1, 1, 1],
                     padding='VALID')
conv2 = tf.nn.bias_add(conv2, conv2_b, name='conv2')
conv2 = tf.nn.relu(conv2)

# second pool layer,output 5x5x16
pool2 = tf.nn.max_pool(conv2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID')
# flatten
flatten = tf.contrib.layers.flatten(pool2)
# first full layer
full1_W = tf.Variable(tf.truncated_normal([400, 120], mu, sigma))
full1_b = tf.Variable(tf.truncated_normal([120], mu, sigma))
full1 = tf.add(tf.matmul(flatten, full1_W), full1_b)

full1 = tf.nn.relu(full1)
full1 = tf.nn.dropout(full1, keep_prob)

# second full layer
full2_W = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))
full2_b = tf.Variable(tf.truncated_normal([84], mu, sigma))
full2 = tf.add(tf.matmul(full1, full2_W), full2_b)

full2 = tf.nn.relu(full2)
full2 = tf.nn.dropout(full2, keep_prob)

# third full layer
full3_W = tf.Variable(tf.truncated_normal([84, n_classes], mu, sigma))
full3_b = tf.Variable(tf.truncated_normal([n_classes], mu, sigma))
logits = tf.add(tf.matmul(full2, full3_W), full3_b, name='final')

# model finish

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)

learning_rate = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate)
train_operation = optimizer.minimize(loss)

####
predict_result = tf.argmax(logits, 1)
correct_prediction = tf.equal(predict_result, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(session, examples, labels, batch=100):
    num_examples = len(examples)
    accuracy_total = 0
    loss_total = 0
    for offset in range(0, num_examples, batch):
        batch_x, batch_y = examples[offset:offset + batch], labels[offset:offset + batch]
        loss_batch, accuracy_batch = session.run([loss, accuracy_operation],
                                                 feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy_total += (accuracy_batch * batch)
        loss_total += loss_batch * batch
    return loss_total / num_examples, accuracy_total / num_examples


####
import time


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('function {}\'s running time is: {}'.format(func.__name__, end - start))

    return wrapper


@timing
def train_model(rate, batch_size, epoch, save=False):
    print('rate={},batch_size={},epoch={}'.format(rate, batch_size, epoch))
    batches_valid_accuracy = []
    batches_train_accuracy = []
    batches_train_losses = []
    batches_valid_losses = []
    batch_indexs = []
    rescent_train_accuracy = []
    rescent_valid_accuracy = []

    stop_training_threshold = 0.005
    log_stride = 50
    saver = tf.train.Saver()
    plt.subplots_adjust()

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        num_examples = len(b_train_features)

        print('Training...')
        for i in range(epoch):
            shuffled_train_features, shuffled_train_labels = shuffle(b_train_features, b_train_labels)
            batch_position = 0
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_position += 1
                batch_samples = shuffled_train_features[offset:end]
                batch_labels = shuffled_train_labels[offset:end]

                session.run(train_operation,
                            feed_dict={X: batch_samples, y: batch_labels, learning_rate: rate, keep_prob: 0.75})

                if batch_position % log_stride == 0:
                    loss, accuracy = evaluate(session, b_train_features, b_train_labels, batch_size)
                    batches_train_accuracy.append(accuracy)
                    batches_train_losses.append(loss)
                    loss, accuracy = evaluate(session, b_valid_features, b_valid_labels, batch_size)
                    batches_valid_accuracy.append(accuracy)
                    batches_valid_losses.append(loss)

                    last_size = batch_indexs[-1] if batch_indexs else 0
                    batch_indexs.append(last_size + log_stride * batch_size)

            _, valid_accuracy = evaluate(session, b_valid_features, b_valid_labels, batch_size)

            _, train_accuracy = evaluate(session, b_train_features, b_train_labels, batch_size)
            if len(rescent_train_accuracy) > 0 and abs(np.mean(
                    np.array(rescent_train_accuracy[-5:])) - train_accuracy) < stop_training_threshold and \
                            abs(np.mean(
                                np.array(rescent_train_accuracy[-5:])) - train_accuracy) < stop_training_threshold:
                break
            rescent_train_accuracy.append(train_accuracy)
            rescent_valid_accuracy.append(valid_accuracy)

            print('Epoch ', i)
            print('train accuracy is: {}'.format(train_accuracy))
            print('validation accuracy is: {}'.format(valid_accuracy))

        plt.subplot(2, 1, 1)
        plt.title('Accuracy')
        plt.plot(batch_indexs, batches_train_accuracy, 'r', label='Train accuracy')
        plt.plot(batch_indexs, batches_valid_accuracy, 'x', label='Valid accuracy')
        plt.subplot(2, 1, 2)
        plt.title('Loss')
        plt.plot(batch_indexs, batches_train_losses, 'b')
        plt.plot(batch_indexs, batches_valid_losses, 'gx')

        plt.savefig('LeNet_%s_%d_%d_%s' % (str(rate)[2:], batch_size, epoch, str(keep_prob)[2:]))
        # plt.show()

        print('train accuracy is: {}'.format(train_accuracy))
        print('validation accuracy is: {}'.format(valid_accuracy))
        #         assert valid_accuracy > 0.93 'less than 0.93'

        if save:
            if not os.path.exists('model'):
                os.makedirs('model')
            saver.save(session, './model/classifier')
            print('saved')


# for i in [0.001, 0.0008, 0.0005]:
#     train_model(i, 100, 30)
#
# for i in [50, 100, 200]:
#     train_model(0.001, i, 30)

# train_model(0.001, 50, 30,True)
# train_model(0.0008, 50, 50)


####
with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session, './model/classifier')
    x_test_features = normalize(test_features)
    print('Test accuracy is :', evaluate(session, x_test_features, test_labels)[1])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path as path
import csv

# reader() return _csv.reader object
classes_map = {}
with open('signnames.csv', mode='r') as f:
    signnames_reader = csv.reader(f)
    signnames_reader.__next__()
    for row in signnames_reader:
        classes_map[row[0]] = row[1]


def class_name(i):
    if str(i) in classes_map:
        return classes_map[str(i)]


all_signs = []
for i in range(n_classes):
    i_index = test_labels.tolist().index(i)
    all_signs.append(test_features[i_index])

pic_per_row = 4
_, axes = plt.subplots(11, pic_per_row, figsize=(12, 20))
for i, j in enumerate(all_signs):
    axes[i // pic_per_row, i % pic_per_row].imshow(j)
    axes[i // pic_per_row, i % pic_per_row].set_title(str(i) + ':' + class_name(i))

plt.tight_layout(0, 0.3, 0.2)
plt.savefig('output_images/show_traffic_signs.png')
plt.show()

import matplotlib.image as mpimg
import glob

fnames = glob.glob('new_images/*')

new_test_imgs = []

for fname in fnames:
    new_test_imgs.append(mpimg.imread(fname))

new_test_labels = [12, 14, 17, 35, 1]

plt.title('New Sign')
for i in range(len(new_test_imgs)):
    plt.subplot(2, 3, i + 1)
    plt.title(fnames[i].split('/')[1])
    plt.imshow(new_test_imgs[i])
plt.tight_layout()
plt.savefig('output_images/new_test_images.png')
plt.show()

from skimage.transform import resize
import numpy as np

resized_test_imgs = []
for i in new_test_imgs:
    resized_test_imgs.append(cv2.resize(i, (32, 32)))

resized_test_imgs = np.array(resized_test_imgs)

resized_test_imgs = normalize(resized_test_imgs).astype(np.float32)


def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    #     print(image_input)
    print(tf_activation)
    activation = tf_activation.eval(session=session, feed_dict={X: image_input, keep_prob: 1.0})
    print(activation.shape)
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 9))
    for featuremap in range(featuremaps):
        plt.subplot(3, 6, featuremap + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                       vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
    plt.tight_layout(0, 0.02, 0.02)
    #     plt.subplots_adjust()
    plt.savefig('output_images/visulize_cnn.png')
    plt.show()


with tf.Session() as session:
    saver.restore(session, './model/classifier')
    outputFeatureMap(resized_test_imgs[0:1], session.graph.get_tensor_by_name('conv2:0'))
