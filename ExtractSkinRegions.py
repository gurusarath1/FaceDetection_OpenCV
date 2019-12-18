import cv2
import numpy as np

def extractSkinRegions(image):

	GrayScale_Image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
	
	SkinDetectedImage = image.copy();

	i = 0
	j = 0
	for px in SkinDetectionMask:

		j = 0
		for py in px:
			if py == 0:
				SkinDetectedImage[i][j] = 0
			j += 1
			
		i += 1
		
	cv2.imshow("Original Image", image)
	SkinDetectionMask = np.float32(SkinDetectionMask)
	cv2.imshow("Skin Detection Mask", SkinDetectionMask)
	cv2.imshow("Skin Detected image", SkinDetectedImage)
	cv2.waitKey(0)
	return SkinDetectedImage
	
extractSkinRegions(cv2.imread(".\\images\\zac.jpg"))