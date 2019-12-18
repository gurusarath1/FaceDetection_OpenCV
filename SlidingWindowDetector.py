import numpy as np
import imutils
import time
import cv2

def ScaleImage(InputImage, minImageSize, scaleFactor=1.5):

	yield InputImage

	# Loop until the image becomes very small
	while True:
		# compute the new dimensions of the InputImage and resize it
		w = int(InputImage.shape[1] / scaleFactor)
		InputImage = imutils.resize(InputImage, width=w)

		if InputImage.shape[0] < minImageSize[1] or InputImage.shape[1] < minImageSize[0]:
			break

		# Next Image
		yield InputImage

def generateNextWindowLocation(InputImage, boxSize, stride):
	# slide a window across the InputImage
	for y in range(0, InputImage.shape[0], stride):
		for x in range(0, InputImage.shape[1], stride):
			# Next Window
			yield (x, y, InputImage[y:y + boxSize[1], x:x + boxSize[0]])

# Load the image
imageFolder = ".\\images\\"
fileName = "3.jpg"
image = cv2.imread(imageFolder + fileName)
#Load the model file
modelFolder = '.\\xmls\\'
modelFile = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(modelFolder + modelFile)

# Create a copy of image to overlay the window rectangle
ImageCopy = image.copy()

#Window Dimensions
(winW, winH) = (50, 50)

# loop over the image pyramid

i = 0
skipNum = 2
for resized in ScaleImage(image, scaleFactor=1.2, minImageSize=(30,30)):

	# This if statement is used to quickly resize the image
	if i < skipNum:
		i = i+1
		continue

	# Scan all regions of the image
	for (x, y, window) in generateNextWindowLocation(resized, boxSize=(winW, winH), stride=5):
		
		
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# Get the image within the rectangle
		gray_cropped = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
		CroppedImg_shape = gray_cropped.shape
		cv2.imshow("Window - Cropped", gray_cropped)
		
		p = face_cascade.detectMultiScale(gray_cropped, scaleFactor = 1.01)
		
		# If face was detected
		# Wait for 2 seconds
		if len(p) > 0:
			print("Face Detected in this region", print(p))
			time.sleep(2)

		#Draw bounding Box
		ImageCopy = resized.copy()
		leftLoc = (x, y)
		RightLoc = (x + winW, y + winH)
		cv2.rectangle(ImageCopy, leftLoc, RightLoc, (0, 255, 0), 2)
		cv2.imshow("Image", ImageCopy)
		cv2.waitKey(1)
		

