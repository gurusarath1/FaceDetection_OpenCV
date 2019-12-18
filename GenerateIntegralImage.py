import numpy as np
import cv2

def integralImage(image):
	#Ensure that the image is grayscale image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Get the dimensions of the input image
	imageShape = image.shape
	#Output image
	out = np.zeros(imageShape)
	
	x = 0
	for i in range(imageShape[0]):
		y = 0
		for j in range(imageShape[1]):
			#print(image[:x,:y])
			out[x,y] = np.sum(image[:x+1,:y+1])
			y += 1
		x = x + 1
		
	return out