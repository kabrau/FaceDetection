import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from detectors import CascadeDetector
from detectors import SSDDetector
from detectors import ImageTools 

imageSize = None

detectorCascade = CascadeDetector()
detectorCascade.imageSize = imageSize
detectorCascade.Open()

detectorSSD = SSDDetector()
detectorSSD.imageSize = imageSize
detectorSSD.Open()

tools = ImageTools()

imagesFolder = "./samples"
valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg"]


for filename in os.listdir(imagesFolder):     

     ext = os.path.splitext(filename)[1]
     if ext.lower() not in valid_images:
          continue

     fileNamePath = os.path.join(imagesFolder, filename)
     test_image = cv2.imread(fileNamePath)
     #test_image = imutils.resize(test_image, width=640)

     boxesCascade = detectorCascade.Detect(test_image)
     boxesSSD = detectorSSD.Detect(test_image)

     test_image = imutils.resize(test_image, width=640)

     tools.PrintBoxes(test_image, boxesCascade,  (0,255,0), "Cascade" )
     tools.PrintBoxes(test_image, boxesSSD, (255,0,0), "SSD" )
     
     cv2.imshow("Frame", test_image)

     plt.show()
     key = cv2.waitKey(3000) & 0xFF
 