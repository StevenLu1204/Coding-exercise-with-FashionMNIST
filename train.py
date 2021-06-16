import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from tensorflow.keras import backend as K
import os

def create_model():

# load MobileNet v2 and construct a classifier

	mbnet = tf.keras.applications.MobileNet(
		input_shape = (56, 56, 3),
		include_top = False,
		pooling = 'avg')

	pred_layer = tf.keras.layers.Dense(10, activation='softmax')

	inputs = tf.keras.Input(shape = (56, 56))
	input_image_ = tf.keras.layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x,3),3,3))(inputs)
	x = mbnet(input_image_)
	x = tf.keras.layers.Dropout(0.5)(x)
	outputs = pred_layer(x)
	model = tf.keras.Model(inputs, outputs)

	model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
		loss = 'sparse_categorical_crossentropy',
		metrics = ['accuracy'])
	model.summary()

	return model


def main():

	# load training and testing data from fashion mnist

	fashion_mnist = tf.keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# resize the images to 56x56

	height, width = 56, 56

	train_images = train_images.reshape((-1,28,28))
	train_images = np.array([resize(x, (height,width)).astype(float) for x in tqdm(iter(train_images))])/255.

	test_images = test_images.reshape((-1,28,28))
	test_images = np.array([resize(x, (height,width)).astype(float) for x in tqdm(iter(test_images))])/255.

	print(train_images.shape)

	model = create_model()

	weight_path = "./weights-3/cp-{epoch:04d}.ckpt"

	batch_size = 128

	callback = tf.keras.callbacks.ModelCheckpoint(
		filepath = weight_path,
		verbose = 1,
		save_weights_only = True,
		save_freq = 5*batch_size)

	history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=50, callbacks = [callback], validation_data = (test_images, test_labels))

if __name__ == '__main__':
	main()