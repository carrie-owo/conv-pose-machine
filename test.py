import numpy as np
import tensorflow as tf
import cv2
import os
from model import CPMModel
from tensorflow import keras
import seaborn as sns
from gen_data import gaussian_kernel
import matplotlib.pyplot as plt

image_shape = (368,368,3)

learning_rate = 1e-3
optimizer = keras.optimizers.Adam(learning_rate)


def get_key_points(heatmap, height, width):
	"""
	Get all key points from heatmap6.
	:param heatmap6: The heatmap6 of CPM cpm.
	:param height: The height of original image.
	:param width: The width of original image.
	:return: All key points of the person in the original image.
	"""
	key_points = []
	# Get k key points from heatmap6
	for x in range(1,15):
		maxv,maxi,maxj=0,0,0
		for i in range(46):
			for j in range(46):
				if heatmap[i,j,x]>maxv:
					maxv,maxi,maxj=heatmap[i,j,x],i,j
		key_points.append([maxi*8, maxj*8])

	return key_points


def draw_image(image, key_points):
	"""
	Draw limbs in the image.
	:param image: The test image.
	:param key_points: The key points of the person in the test image.
	:return: The painted image.
	"""
	'''ALl limbs of person:
	head top->neck
	neck->left shoulder
	left shoulder->left elbow
	left elbow->left wrist
	neck->right shoulder
	right shoulder->right elbow
	right elbow->right wrist
	neck->left hip
	left hip->left knee
	left knee->left ankle
	neck->right hip
	right hip->right knee
	right knee->right ankle
	'''
	limbs = [[13, 12], [12, 9], [9, 10], [10, 11], [12, 8], [8, 7], [7, 6], [12, 3], [3, 4], [4, 5], [12, 2], [2, 1],
	         [1, 0]]

	# draw key points
	for key_point in key_points:
		x = key_point[0]
		y = key_point[1]
		cv2.circle(image, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

	# draw limbs
	for limb in limbs:
		start = key_points[limb[0]]
		end = key_points[limb[1]]
		color = (0, 0, 255)  # BGR
		cv2.line(image, tuple(start), tuple(end), color, thickness=1, lineType=4)

	return image


if __name__ == "__main__":
	latest = tf.train.latest_checkpoint("./ck3/")
	image_path = 'a.jpg'

	image = cv2.imread(image_path)

	height, width, _ = image.shape
	image = np.asarray(image, dtype=np.float32)
	image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)

	# Normalize
	image -= image.mean()

	# Generate center map
	centermap = np.zeros((368, 368, 1), dtype=np.float32)
	kernel = gaussian_kernel(size_h=368, size_w=368, center_x=184, center_y=184, sigma=3)
	kernel[kernel > 1] = 1
	kernel[kernel < 0.01] = 0
	centermap[:, :, 0] = kernel
	centermap = tf.convert_to_tensor(np.transpose(centermap, (2, 0, 1)))
	
	image = tf.expand_dims(image, axis=0)
	centermap = tf.reshape(centermap, [1,368,368,1])

	inputs = keras.Input(shape=image_shape)
	network = CPMModel()
	
	network.load_weights(latest)

	heat1, heat2, heat3, heat4, heat5, heat6 = network(image, centermap)
	
	heat = heat6[0,:,:, :]
	for i in range(15):
		heat = heat6[0,:,:, i]
		print(heat)
		plt.figure()
		p1 = sns.heatmap(heat)

		figure = p1.get_figure()
		try:
			figure.savefig("ans" + str(i) + ".png", dpi=400)
			print("ok:",i)
		except:
			print("error",i)

	# key_points = get_key_points(heat6, height=height, width=width)

	# image = draw_image(cv2.imread(image_path), key_points)

	# cv2.imshow('test image', image)
	# cv2.waitKey(0)
