#https://www.datacamp.com/community/tutorials/face-detection-python-opencv
#https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# %matplotlib inline

def convertToRGB(image):
     return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#Loading the image to be tested
#test_image = cv2.imread('E:/datasets/signal/valid/Img 1863-6-(F00154).jpeg')
#test_image = cv2.imread('E:/datasets/signal/valid/20170815_124737-(F00112).jpeg')
PATH_TO_TEST_IMAGES_DIR = 'E:/datasets/signal/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'valid/20170815_124737-(F00112).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'valid/Img 1863-6-(F00154).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test/20170812_171208-(F00075).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test/Img 1860-3-(F00261).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test/VID_20170812_150111228_HDR-(F00165).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test/VID_20170812_170449656_HDR-(F00098).jpeg'),
                     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test/WhatsApp Video 2017-12-02 at 6.13.33 PM-(F00190).jpeg'),                    
                   ]

for filenameImage in TEST_IMAGE_PATHS:
     test_image = cv2.imread(filenameImage)

     #Converting to grayscale
     test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

     # Displaying the grayscale image
     plt.imshow(test_image_gray, cmap='gray')
     # Since we know that OpenCV loads an image in BGR format, so we need to convert it into RBG format to be able to display its true colors. Let us write a small function for that.
     #plt.show()


     #haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
     haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

     faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.1, minNeighbors = 5)

     # Let us print the no. of faces found
     print('Faces found: ', len(faces_rects))

     for (x,y,w,h) in faces_rects:
          cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

     #convert image to RGB and show image

     plt.imshow(convertToRGB(test_image))

     plt.show()