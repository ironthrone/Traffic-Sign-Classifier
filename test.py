import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import os.path as path
import csv
import cv2
import tensorflow as tf



def normalize(features):
    max = 255
    min = 0
    high_bound = 1
    low_bound = -1
    return low_bound + ((features - min) / (max - min)) * (high_bound - low_bound)

import matplotlib.image as mpimg
import glob
fnames = glob.glob('new_images/*')

new_test_imgs = []

for fname in fnames:
    new_test_imgs.append(mpimg.imread(fname))

new_test_labels = [12,14,17,35,1]

plt.title('New Sign')
for i in range(len(new_test_imgs)):
    plt.subplot(2,3,i+1)
    plt.title(fnames[i].split('/')[1])
    plt.imshow(new_test_imgs[i])
plt.tight_layout()
plt.savefig('output_images/new_test_images.png')
plt.show()

from skimage.transform import resize
import numpy as np

resized_test_imgs = []
for i in new_test_imgs:
    resized_test_imgs.append(cv2.resize(i,(32,32)))

resized_test_imgs = np.array(resized_test_imgs)

resized_test_imgs = normalize(resized_test_imgs).astype(np.float32)


X = tf.placeholder(tf.float32,(None,32,32,3))
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    #     print(image_input)
    print(tf_activation)
    activation = tf_activation.eval(session=session, feed_dict={X: image_input})
    print(activation.shape)
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(20, 12))
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


with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session, './model/classifier')
    print(resized_test_imgs[0:1].dtype)
    outputFeatureMap(resized_test_imgs, session.graph.get_tensor_by_name('conv2:0'))