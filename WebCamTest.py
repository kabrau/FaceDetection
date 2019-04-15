#https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

# import the necessary packages
from imutils.video import VideoStream
import time
import cv2
import os
import imutils
from detectors import CascadeDetector
from detectors import SSDDetector
from detectors import ImageTools 

#===========================
useCascade = True
useSSD = True
#===========================

if useCascade:
	detectorCascade = CascadeDetector()
	detectorCascade.Open()

if useSSD:
	detectorSSD = SSDDetector()
	detectorSSD.Open()

tools = ImageTools()

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

outputFolder = ".\output\\"

# loop over the frames from the video stream
while True:
	original_frame = vs.read()
	frame = original_frame.copy()
 
	if useCascade:
		boxesCascade = detectorCascade.Detect(frame)

	if useSSD:
		boxesSSD = detectorSSD.Detect(frame)
	
	if useCascade:
		tools.PrintBoxes(original_frame, boxesCascade,  (0,255,0), "Cascade" )
	if useSSD:
		tools.PrintBoxes(original_frame, boxesSSD, (255,0,0), "SSD" )
 
	# show the output frame
	original_frame = imutils.resize(original_frame, width=1080)
	cv2.imshow("Frame", original_frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `o` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("o"):
		p = os.path.sep.join([outputFolder, "{}.png".format( str(total).zfill(5))])
		cv2.imwrite(p, original_frame)
		total += 1

	# if the `d` key was pressed, write the *detected* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("d"):
		p = os.path.sep.join([outputFolder, "{}_detected.png".format( str(total).zfill(5))])
		cv2.imwrite(p, frame)
		total += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break        

# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()