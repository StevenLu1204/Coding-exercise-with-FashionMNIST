import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from keras import backend as K
import cv2
import random

random.seed(10)

def create_model():
	# create MobileNet v2 with ImageNet weight

	mbnet = tf.keras.applications.MobileNet(
	input_shape = (56, 56, 3),
	include_top = False,
	pooling = 'avg')

	pred_layer = tf.keras.layers.Dense(10, activation='softmax')

	inputs = tf.keras.Input(shape = (56, 56))
	input_image_ = tf.keras.layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(inputs) # expand channel size to 3
	x = mbnet(input_image_)
	x = tf.keras.layers.Dropout(0.5)(x)
	outputs = pred_layer(x)
	model = tf.keras.Model(inputs, outputs)

	model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy'])

	return model

def cal_accuracy(images, labels):
	# calculate the accuracy of the testing images

	model = create_model()
	model.load_weights("./weights/cp-0049.ckpt")

	loss, acc = model.evaluate(images, labels, verbose=2)
	print("Accuracy: {:5.2f}%".format(100 * acc))

	return acc

def complement_image(images, percent):
	# convert % of images to its complementary value

	c_images = images.copy()

	width, height = 28, 28
	
	for x in range(len(images)):
		
		c_images[x][:int(height*percent)] = 255 - images[x][:int(height*percent)]

	return c_images 


def show_image(image):
	# show the image

	plt.imshow(image, cmap=plt.cm.binary)
	plt.show()

	return 0

def main():

	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# Resize the image to 56x56 so it fits the model input shape
	width, height = 56, 56
	test_images = np.array([resize(x, (height,width)).astype(float) for x in tqdm(iter(test_images))]) 

	comp_acc = []
	gblur_acc = []

	# Calculate the accuracy of the processed testing images

	for i in range(0, 101, 5):

		c_images = complement_image(test_images, i*0.01)
		gblur_images = cv2.GaussianBlur(test_images[:], (3,3), i/5)

		c_images = c_images.reshape((-1,56,56))
		gblur_images = gblur_images.reshape((-1,56,56))

		c_acc = cal_accuracy(c_images/255., test_labels)
		g_acc = cal_accuracy(gblur_images/255., test_labels)

		comp_acc.append([i, c_acc])
		gblur_acc.append([i/5, g_acc])

	comp_acc = np.array(comp_acc)
	gblur_acc = np.array(gblur_acc)

	# Plot the figures

	plt.plot(comp_acc[:,0], comp_acc[:,1], 'o-')
	plt.xlabel("complementary percentage")
	plt.ylabel("accuracy")
	plt.savefig("com_acc.png", dpi=150)
	plt.show()

	plt.plot(gblur_acc[:,0], gblur_acc[:,1], 'o-')
	plt.xlabel("gaussian std")
	plt.ylabel("accuracy")
	plt.savefig("gblur_acc.png", dpi=150)
	plt.show()

if __name__ == "__main__":
	main()



