import os
import cv2
import imutils
import tensorflow as tf
import numpy as np
import label_map_util


# transform box to image positions
def getPositions(image, box):
    im_height, im_width, channels = image.shape
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    return (left, right, top, bottom)

# blur rectangle into image
def printBlur(image2, box, nivelBlur):
    (left, right, top, bottom) = getPositions(image2, box)
    roi_corners = np.array([[(left,top),(right,top),(right,bottom),(left,bottom)]],dtype = np.int32)
    blurred_image = cv2.GaussianBlur(image2,(nivelBlur,nivelBlur),cv2.BORDER_DEFAULT)
    mask = np.zeros(image2.shape, dtype=np.uint8)
    channel_count = image2.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
    final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image2, mask_inverse)
    np.copyto(image2, final_image)

class ImageTools:
    def PrintBoxes(self, image, boxes, color, title):
        if color==None:
            color = (255,0,0)

        for box in boxes:
            (left, right, top, bottom) = getPositions(image, box)
            cv2.rectangle(image, (int(left),int(top)), (int(right),int(bottom)), color, 1)
            if (title != None):
                cv2.putText(image, title, (int(left),int(top)-1) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def BlurBoxes(self, image, boxes):
        for box in boxes:
            printBlur(image, box, 13)

class CascadeDetector:
    
    def __init__(self):
        self.fileModel = 'Cascade/data/haarcascade_frontalface_alt.xml'
        self.detector = None
        self.imageSize = 600

    def Open(self):
        # load OpenCV's Haar cascade for face detection from disk
        self.detector = cv2.CascadeClassifier(self.fileModel)

    def Detect(self, image):
        if (self.imageSize != None):
            frame = imutils.resize(image, width=self.imageSize)
        else:
            frame = image.copy()
        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # transform relative position
        im_height, im_width, channels = frame.shape
        boxes = [[y/im_height, x/im_width, (y+h)/im_height, (x+w)/im_width] for x, y, w, h in rects]
        return boxes
    
class SSDDetector:
    def __init__(self):
        self.detector = None
        self.imageSize = 600
        self.min_score_thresh = 0.5

        # What model to download.
        self.MODEL_NAME = 'SSD/' 
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = self.MODEL_NAME + 'inference/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(self.MODEL_NAME, 'config/classes.pbtxt') 
        self.NUM_CLASSES = 1

    def Open(self):
        #Loading label map
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
 
        #Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        detection_graph.as_default()
        self.detector = tf.Session(graph=detection_graph, config=config)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def Detect(self, image):
        if (self.imageSize != None):
            frame = imutils.resize(image, width=self.imageSize)
        else:
            frame = image.copy()
        
        image_np_expanded = np.expand_dims(frame, axis=0)
        
        # Actual detection.
        (boxes, scores, classes, num) = self.detector.run( 
				[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
				feed_dict={self.image_tensor: image_np_expanded})

        bb = []
        for i in range(boxes.shape[0]):
            if scores[0][i] >= self.min_score_thresh:
                bb.append(boxes[0][i])

        return bb
    
