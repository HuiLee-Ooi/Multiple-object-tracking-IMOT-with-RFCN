import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('/home/huooi/HL_Dataset/UrbanTracker/rouen_video.avi')
# input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_video.avi'
input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
cap = cv2.VideoCapture(input_file)

# cap = cv2.VideoCapture ('/home/huooi/HL_Dataset/CDNet/dataset2014/dataset/baseline/pedestrians/input/in%06d.jpg')
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")



# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'



# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/rfcn_resnet101_coco_2017_11_08/saved_model/saved_model.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'

# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/ssd_inception_v2_coco_2017_11_17/saved_model/saved_model.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb'

# COCO pretrained weight
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/faster_rcnn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mscoco_label_map.pbtxt'
# NUM_CLASSES = 90

# MIO finetuned weight from COCO
# PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/MIO/mio-ssd-inc/train/output_inference_graph3.pb'
# # PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/MIO/faster_rcnn_101/train/output_inference_graph2.pb'
# PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_rfcn_resnet101_mio/train/output_inference_graph.pb'
PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_rfcn_resnet101_mio/frozen/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mio_label_map.pbtxt'
NUM_CLASSES = 11

# PET finetuned weight from COCO
# PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_test_faster_rcnn_resnet101_pets/train/output_inference_graph.pb'
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/pet_label_map.pbtxt'
# NUM_CLASSES = 37

# KITTI pretrained weight
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/faster_rcnn_resnet101_kitti_2017_11_08/frozen_inference_graph.pb'
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/kitti_label_map.pbtxt'
# NUM_CLASSES = 8

# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
# ## Load a (frozen) Tensorflow model into memory.

# In[6]:
nms_thres = 0.3

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
#
# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
    # while (cap.isOpened()):
      ret, image_np = cap.read()
      imagecopy = image_np.copy()
      if not ret:
          # just wanna read the operation names available
        # for op in detection_graph.get_operations():
        #     print str(op.name)
        break

      height, width = image_np.shape[:2]
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2,
      )

      final_score = np.squeeze(scores)
      count = 0
      box_nms_block = []
      centers = []
      color_per_frame = []
      pos_per_frame = []
      hist_per_frame = []
      label_per_frame = []
      for i_init in range((np.squeeze(num_detections))):
        # if scores is None or final_score[i] > 0.5:
        if (scores is None or final_score[i_init] > 0.5) and (
            (np.reshape(classes, int(np.squeeze(num_detections)), 1)[i_init]) != 7.0):
            count = count + 1
            ymin_old = int(np.round(np.squeeze(boxes)[i_init, 0] * height))
            xmin_old = int(np.round(np.squeeze(boxes)[i_init, 1] * width))
            ymax_old = int(np.round(np.squeeze(boxes)[i_init, 2] * height))
            xmax_old = int(np.round(np.squeeze(boxes)[i_init, 3] * width))
            lbl_type = np.reshape(classes, int(np.squeeze(num_detections)), 1)[i_init]
            lbl_conf = final_score[i_init] * 100
            box_nms = [xmin_old, xmax_old, ymin_old, ymax_old, lbl_type, lbl_conf]
            box_nms_block.append(box_nms)

      pick = non_max_suppression_fast(np.array(box_nms_block), nms_thres)

      for i in range(len(pick)):
        if (True):  # added just because i dont wanna change the alignment of everything
            xmin = (pick[i][0])
            xmax = (pick[i][1])
            ymin = (pick[i][2])
            ymax = (pick[i][3])
            cv2.rectangle(imagecopy, (xmin, ymax), (xmax, ymin), (255, 0, 0), 2)
      # print (np.squeeze(classes).astype(np.int32))

    # for i in range((np.squeeze(num_detections))):
    #     if scores is None or final_score[i] > 0.5:
    #         count = count + 1
    #         ymin = int(np.round(np.squeeze(boxes)[i, 0] * im_height))

      # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      # print (count)
      # print (len(pick))
      if (count!=len(pick)):
          print ("check this out!")
      cv2.imshow('nms', cv2.resize(imagecopy, (width, height)))
      cv2.imshow('original', cv2.resize(image_np, (width, height)))
      cv2.waitKey(0)
      # if cv2.waitKey(25) & 0xFF == ord('q'):
      #   cv2.destroyAllWindows()
      #   break
