import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from scipy.optimize import linear_sum_assignment
from multitracker import Tracker
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



from xml.etree.ElementTree import Element, SubElement, Comment, ElementTree, XML
import datetime
import csv
from xml.dom import minidom
# from xml.etree import ElementTree
from xml.etree import ElementTree as ET


# import sys
# sys.path.append('/usagers2/huooi/dev/HL_Proj/opencv/build/lib')
# print (sys.path)

import cv2

# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_video.avi')
# input_file = '/home/huooi/HL_Dataset/UrbanTracker/dummy_folder/mac18/%08d.jpg'

nms_thres = 0.3
# nms_thres = 0.15

input_mode = "sherbrooke" #"sherbrooke","rouen","rene","stmarc"
ope_mode = "wo_label" #"w_label", "wo_label"
if (input_mode == "sherbrooke"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
    gt_st_frame = 2754
    gt_ed_frame = 3754
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/sherbrooke_w_l.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/sherbrooke_wo_l.xml'
elif (input_mode == "rouen"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rouen_frames/%08d.jpg'
    gt_st_frame = 20
    gt_ed_frame = 620
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/rouen_w_l.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/rouen_wo_l.xml'
elif (input_mode == "rene"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rene_frames/%08d.jpg'
    gt_st_frame = 7200
    gt_ed_frame = 8199
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/rene_w_l.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/rene_wo_l.xml'
elif (input_mode == "stmarc"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/stmarc_frames/%08d.jpg'
    gt_st_frame = 1000
    gt_ed_frame = 1999
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/stmarc_w_l.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/stmarc_wo_l.xml'
else:
    print ("no file chosen")

# input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
# output_file = '/home/huooi/HL_Results/MOT_xml/xml_sherbrooke'


# input_file = '/usagers/huooi/dev/HL_Dataset/UrbanTracker/dum2/%08d.jpg'
# /home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames
# output_file = '/usagers/huooi/dev/HL_Proj/PycharmProjects/models/object_detection/result/xml_sherbrooke'


try:
    os.remove(output_file)
except OSError:
    pass

cap = cv2.VideoCapture(input_file)
# cap = cv2.VideoCapture('/home/huooi/HL_Dataset/UrbanTracker/dummy_folder/new_dum/%08d.jpg')


# cap = cv2.VideoCapture ('/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg')

import sqlite3
# sqlite_file = '/usagers/huooi/dev/HL_Dataset/UrbanTracker/Annotations/sherbrooke_annotations/sherbrooke_gt.sqlite'
# conn = sqlite3.connect(sqlite_file)
# c = conn.cursor()
# res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
# for name in res:
#     print name[0]

sqlite_file = '/usagers2/huooi/dev/HL_Dataset/UrbanTracker/Annotations/sherbrooke_annotations/sherbrooke_gt.sqlite'
conn = sqlite3.connect(sqlite_file)

cursor = conn.cursor()
cursor.execute("SELECT min(frame_number) FROM bounding_boxes")
for row in cursor:
    min_frame_count = row[0]
cursor.execute("SELECT max(frame_number) FROM bounding_boxes")
for row in cursor:
    max_frame_count = row[0]

cursor.execute("SELECT * FROM bounding_boxes")
cursor2 = conn.cursor()
cursor3 = conn.cursor()

# cap = cv2.VideoCapture ('/home/huooi/HL_Dataset/CDNet/dataset2014/dataset/baseline/pedestrians/input/in%06d.jpg')
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# # Model preparation
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# COCO pretrained weight
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/Pretrained models/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/Pretrained models/rfcn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/Pretrained models/faster_rcnn_resnet101_coco_2017_11_08/frozen_inference_graph.pb'
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mscoco_label_map.pbtxt'
# NUM_CLASSES = 90

# MIO finetuned weight from COCO
PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/HL_rfcn_resnet101_mio/frozen/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mio_label_map.pbtxt'
NUM_CLASSES = 11

# PET finetuned weight from COCO
# PATH_TO_CKPT = ''
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/pet_label_map.pbtxt'
# NUM_CLASSES = 37

# KITTI pretrained weight
# PATH_TO_CKPT ='/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/Pretrained models/faster_rcnn_resnet101_kitti_2017_11_08/frozen_inference_graph.pb'
# PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/kitti_label_map.pbtxt'
# NUM_CLASSES = 8

# ## Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

# def compute_IOU(Reframe,GTframe):
#     x1 = Reframe[0];
#     y1 = Reframe[1];
#     width1 = Reframe[2]-Reframe[0];
#     height1 = Reframe[3]-Reframe[1];
#
#     x2 = GTframe[0];
#     y2 = GTframe[1];
#     width2 = GTframe[2]-GTframe[0];
#     height2 = GTframe[3]-GTframe[1];
#
#     endx = max(x1+width1,x2+width2);
#     startx = min(x1,x2);
#     width = width1+width2-(endx-startx);
#
#     endy = max(y1+height1,y2+height2);
#     starty = min(y1,y2);
#     height = height1+height2-(endy-starty);
#
#     if width <=0 or height <= 0:
#         ratio = 0
#     else:
#         Area = width*height;
#         Area1 = width1*height1;
#         Area2 = width2*height2;
#         ratio = Area*1./(Area1+Area2-Area);
#     # return IOU
#     return ratio,Reframe,GTframe

def convert_type_to_urbantracker_format(lbl_out):
    # Original label from trained MIO-TCD
    if lbl_out[0] == 1: #articulated truck
        result = ["truck", 6]
    elif lbl_out[0] == 2: #bicycle
        result = ["bicycle",4]
    elif lbl_out[0] == 3: #bus
        result = ["bus",5]
    elif lbl_out[0] == 4: #car
        result = ["car",1]
    elif lbl_out[0] == 5: #motorcycle
        result = ["motorcycle",3]
    elif lbl_out[0] == 8: #pedestrian
        result = ["pedestrian",2]
    elif lbl_out[0] == 9: #pickup truck
        result = ["truck", 6]
    elif lbl_out[0] == 10: #single unit truck
        result = ["truck", 6]
    elif lbl_out[0] == 11: #work van
        result = ["truck", 6]
    elif lbl_out[0] == 6:  # motorized vehicles
        result = ["car", 6]
    else: # including motorized vehicles and non motorized vehicles (for now)
        result = ["unknown",0]
    return result

def convert_bb_points(coord_list):
    """Return a bounding box that conform to groundtruth.
    """
    # x, y are the minimum x and minimum y (which is the bottom left corner of bounding box)
    coord_list = np.array(coord_list)

    x = int(coord_list[2])
    y = int(coord_list[0])
    width = int(coord_list[3] - coord_list[2])
    height = int(coord_list[1] - coord_list[0])

    return [x, y, width, height]


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)

# # Detection
#
#  For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
#
# # Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)
# Malisiewicz et al.
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

frame_num = 1
skipped_frames = 0
max_frame = 5
trackIdCount = 0
feat_per_frame = []
feat_for_each_frame=[]
track = []
path =[]
ls_path =[]
feat = 2
skip_frame_count = 0
# track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),(0, 255, 255), (255, 0, 255),
#                 (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128),
#                 (0, 128, 128), (0, 0, 128), (255, 127, 255),(127, 0, 255), (127, 0, 127), (255,165,0), (0,0,0), (255,255,255),
#                 (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
#                 (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
#                 (255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
# (255, 255, 255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),
#                 ]

# track_colors = [1000] * (255,255,255)

lst = [(255,255,255)]
track_colors = lst * 1000

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# tracker = Tracker(90, 10, 8, 3, height, width)
tracker = Tracker(1.5, 10, 5, 3, height, width, ope_mode)
# tracker = Tracker(1.8, 10, 5, 3, height, width, ope_mode)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
    # while (cap.isOpened()):
      ret, image_np = cap.read()
      if not ret:
        break

      # height, width = image_np.shape[:2]
      # im_height, im_width, channels = image_np.shape

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
      hist_per_frame=[]
      label_per_frame=[]
      for i_init in range((np.squeeze(num_detections))):
        # if scores is None or final_score[i] > 0.5:
        if (scores is None or final_score[i_init] > 0.5) and ((np.reshape(classes,int(np.squeeze(num_detections)),1)[i_init]) !=7.0):
            count = count + 1
            ymin_old = int (np.round(np.squeeze(boxes)[i_init, 0] * height))
            xmin_old = int (np.round(np.squeeze(boxes)[i_init, 1] * width))
            ymax_old = int (np.round(np.squeeze(boxes)[i_init, 2] * height))
            xmax_old = int (np.round(np.squeeze(boxes)[i_init, 3] * width))
            lbl_type = np.reshape(classes, int(np.squeeze(num_detections)), 1)[i_init]
            lbl_conf = final_score[i_init]*100
            box_nms = [xmin_old, xmax_old, ymin_old, ymax_old, lbl_type, lbl_conf]
            box_nms_block.append(box_nms)

      pick = non_max_suppression_fast(np.array(box_nms_block), nms_thres)
      for i in range(len(pick)):
          if (True): #added just because i dont wanna change the alignment of everything
            xmin = (pick[i][0])
            xmax = (pick[i][1])
            ymin = (pick[i][2])
            ymax = (pick[i][3])

            crop_img = image_np[ymin:ymax, xmin: xmax]
            # cv2.imshow("detected", crop_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            center_pt = [(xmin+xmax)/2, (ymin+ymax)/2]

            # histr = cv2.calcHist([crop_img], [1], None, [256], [0, 256])  # only channel 1
            # cv2.normalize(histr, histr, 0, 255, cv2.NORM_MINMAX)
            # histr = np.int32(np.around(histr)).astype('float32')

            # nested_hist = np.empty((0,3), int)
            nested_hist = np.empty((256, 0), int)
            color = ('b', 'g', 'r')
            for j, col in enumerate(color):
                histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
                # cv2.normalize(histr, histr, 0, 255, cv2.NORM_MINMAX)
                # histr = np.int32(np.around(histr)).astype('float32')
                histr = histr.astype('float32')
                histr = histr / ((ymax - ymin) * (xmax - xmin))
                nested_hist = np.concatenate((nested_hist, histr), axis=1)
            nested_hist = nested_hist.reshape(256*3, 1)
            nested_hist = nested_hist.astype('float32')

            hist_per_frame.append(nested_hist)
            color_per_frame.append(histr)
            #pos_per_frame.append([ymin/height,ymax/height,xmin/width,xmax/width]) #noramlize the corners with image size
            pos_per_frame.append([ymin, ymax, xmin, xmax]) # decide not to normalize position with respect to frame size after all
            # label_per_frame.append([np.reshape(classes,int(np.squeeze(num_detections)),1)[i], final_score[i]])
            label_per_frame.append([lbl_type, float(lbl_conf/100)])

            centers.append(center_pt)



    # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      print("Frame " + str(frame_num) + " with "+ str(count) + " objects")
      feat_per_frame = [pos_per_frame, centers, hist_per_frame, label_per_frame]

      # feature_index = 1 #testing with centers point
      # feature_index = 3  # testing with stacked histogram
      feature_index = 0 # bounding box coordinate

      feature_index = [0, 1, 2, 3]
      out = tracker.UpdateTracker(feat_per_frame, feature_index, frame_num)

      for t, trk in enumerate((out)):
        clr = (trk.retrieve_id())
        color = track_colors[clr]
        output_bb = trk.retrieve_state()
        cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[1])), color, 3, 20, 1, 8)
        cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[1])), color, 3, 20, 1, 8)
        cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[0])), color, 3, 20, 1, 8)
        cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[0])), color, 3, 20, 1, 8)

      # if not(frame_num == 0):
      # if (frame_num >= gt_st_frame):
      if ((frame_num >= gt_st_frame) and (frame_num <= gt_ed_frame)):
          if not (os.path.isfile(output_file)):
              # print ("no file")
              video = Element('Video')
              video.attrib['fname'] = input_file  # must be str; cannot be an int
              video.attrib['start_frame'] = str(frame_num)
              video.attrib['end_frame'] = str(frame_num)

              for t, trk in enumerate((out)):
                  output_bb = trk.retrieve_state()
                  # output_bb = trk.retrieve_record()
                  # print (output_bb)

                  traj = Element("Trajectory", obj_id=str(trk.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk.retrieve_type())[0], start_frame=str(frame_num), end_frame=str(frame_num))
                  # SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
                  SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[1]- output_bb[0]),
                             width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]), frame_no=str(frame_num))
                  video.append(traj)
              open(output_file, "w").write(prettify(video))
          else:
              doc = ET.parse(output_file)
              root = doc.getroot()
              for t, trk in enumerate((out)):
                  output_bb = trk.retrieve_state()
                  # output_bb = trk.retrieve_record()
                  # output_bb = convert_bb_points(trk.retrieve_traj())
                  # try:
                  #     output_bb = convert_bb_points(trk.retrieve_traj())
                  # except:
                  #     print ("error here")
                      # print (trk.retrieve_traj())

                  # print (output_bb)
                  if (root.find(".//Trajectory[@obj_id='" + str(trk.retrieve_id()) + "']")) is None:
                      # create traj element and frame subelement, update root attribute
                      # print ("cannot find, gotta create new element")
                      traj = Element("Trajectory", obj_id=str(trk.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk.retrieve_type())[0], start_frame=str(frame_num), end_frame=str(frame_num))
                      # SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
                      SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[1] - output_bb[0]), width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]),
                                 frame_no=str(frame_num))
                      root.append(traj)
                  else:
                      # create frame subelement on the correct traj element, update root and element attribute
                      # print ("there is such element")
                      for node in root.findall(".//Trajectory[@obj_id='" + str(trk.retrieve_id()) + "']"):
                          node.set('end_frame', "'" + str(frame_num) + "'")
                          # SubElement(node, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
                          SubElement(node, 'Frame', contour_pt="0", annotation="0", observation="0",
                                     height=str(output_bb[1] - output_bb[0]), width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]), frame_no=str(frame_num))

              root.set('end_frame', "'" + str(frame_num) + "'")

              for elem in root.iter('*'):
                  if elem.text is not None:
                      elem.text = elem.text.strip()
                  if elem.tail is not None:
                      elem.tail = elem.tail.strip()

              # Now we have a new well-formed XML document. It is not very nicely formatted...
              xml_out = ET.tostring(root)
              # ...so we'll use minidom to make the output a little prettier
              dom = minidom.parseString(xml_out)
              # print dom.toprettyxml()
              open(output_file, "w").write(dom.toprettyxml())

          # if (frame_num >= gt_st_frame):
          # if (frame_num >= 0):
          #   for t, trk in enumerate((out)):
          #       clr = (trk.retrieve_id())
          #       color = track_colors[clr]
          #       output_bb = trk.retrieve_state()
          #       cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[1])), color, 3, 20, 1, 8)
          #       cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[1])), color, 3, 20, 1, 8)
          #       cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[0])), color, 3, 20, 1, 8)
          #       cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[0])), color, 3, 20, 1, 8)
          #       cv2.imwrite("/home/huooi/HL_Results/MOT_result/frame_sherbrooke/both%d.jpg" % frame_num,
          #               cv2.resize(image_np, (int(width), int(height))))  # saving frame
          #       cv2.waitKey(0)



          # # Turn off display for now
          # for t, trk in enumerate((out)):
          #     clr = (trk.retrieve_id())
          #     color = track_colors[clr]
          #     output_bb = trk.retrieve_state()
          #     cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[1])), color, 3, 20, 1, 8)
          #     cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[1])), color, 3, 20, 1, 8)
          #     cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[0])), color, 3, 20, 1, 8)
          #     cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[0])), color, 3, 20, 1, 8)


      # if (frame_num >= gt_st_frame):
      # if ((frame_num >= gt_st_frame) and (frame_num <= gt_ed_frame)):
      #   cv2.imwrite("/home/huooi/HL_Results/MOT_result/frame_rouen_wo_l/both%d.jpg" % frame_num,
      #           cv2.resize(image_np, (int(width), int(height))))  # saving frames

      # cv2.imshow('object detection', cv2.resize(image_np, (int(width), int(height))))
      # cv2.waitKey(0)

      frame_num = frame_num + 1

