import os
import cv2
import numpy as np
import imutils
from detectors import CascadeDetector
from detectors import SSDDetector
from detectors import ImageTools 

# main
def run():

    #===========================
    imagesFolder = ".\\Exemplos\\"
    outputFolder = "E:\\tempo\\test1\\"

    showImage = True
    saveOriginalFile = False

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

    valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg"]

    # loop over the frames from the video stream
    for filename in os.listdir(imagesFolder):

        ext = os.path.splitext(filename)[1]
        if ext.lower() not in valid_images:
            continue

        fileNamePath = os.path.join(imagesFolder, filename)
        image_Original = cv2.imread(fileNamePath)

        image_np = np.copy(image_Original)

        boxesCascade = []
        boxesSSD = []
        if useCascade:
            boxesCascade = detectorCascade.Detect(image_np)

        if useSSD:
            boxesSSD = detectorSSD.Detect(image_np)
        
        if showImage:
            image_np = imutils.resize(image_np, width=640)
            if useCascade:
                tools.BlurBoxes(image_np, boxesCascade )
            if useSSD:
                tools.BlurBoxes(image_np, boxesSSD )
            if useCascade:
                tools.PrintBoxes(image_np, boxesCascade,  (0,255,0), "Cascade" )
            if useSSD:
                tools.PrintBoxes(image_np, boxesSSD, (255,0,0), "SSD" )

        if saveOriginalFile:
            if useCascade:
                tools.BlurBoxes(image_Original, boxesCascade )
            if useSSD:
                tools.BlurBoxes(image_Original, boxesSSD )

        if len(boxesCascade)>0 or len(boxesSSD):
            print(filename)
        
            # show the output frame
            if showImage:
                cv2.imshow("Frame", image_np)
                key = cv2.waitKey(5000) #& 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            if saveOriginalFile:
                cv2.imwrite(os.path.join(outputFolder, filename),image_Original)

    print("Finish")


#=============================================================================
run()
