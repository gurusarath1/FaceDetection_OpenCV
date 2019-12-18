import numpy as np
import cv2

# Import a image
image = cv2.imread(".\\images\\3.jpg")
# Load the model
face_cascade = cv2.CascadeClassifier('.\\xmls\\haarcascade_frontalface_default.xml')

ImageClone1 = image.copy()

#Get the dimentions of the input image
img_shape = image.shape

GrayScale_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
T,Thresholded_Image = cv2.threshold(GrayScale_Image, 180,255, cv2.THRESH_BINARY)

#Create the YUV image for extraction of skin color
YUV_Image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#Gray scale image
Y_image = YUV_Image[:,:,0]
#Cb Channel
Cb_image = YUV_Image[:,:,1]
#Cr channel
Cr_image = YUV_Image[:,:,2]

Cb_image_Thresholded = (Cb_image > 77) & (Cb_image < 127)
Cr_image_Thresholded = (Cr_image > 133) & (Cr_image < 173)
SkinDetectionMask = Cb_image_Thresholded & Cr_image_Thresholded


SkinDetectedImage = GrayScale_Image.copy();

i = 0
j = 0
for px in SkinDetectionMask:

	j = 0
	for py in px:
		if py == 0:
			SkinDetectedImage[i][j] = 0
		j += 1
		
	i += 1


Faces = face_cascade.detectMultiScale(Thresholded_Image)

for x,y,winW,winH in Faces:
	cv2.rectangle(ImageClone1, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
	

cv2.imshow("Original Image", ImageClone1)
cv2.imshow("GrayScale_Image Image", GrayScale_Image)
#cv2.imshow("Thresholded Image", Thresholded_Image)
#cv2.imshow("YUV Image", YUV_Image)
SkinDetectionMask = np.float32(SkinDetectionMask)
cv2.imshow("Skin Detection", SkinDetectionMask)
cv2.imshow("Skin Detected Image", SkinDetectedImage)
cv2.waitKey(0)