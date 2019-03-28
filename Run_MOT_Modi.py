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
import itertools


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

input_mode = "rouen" #"sherbrooke","rouen","rene","stmarc"
ope_mode = "w_label" #"w_label", "wo_label"
# sz_imot_percent = 0.0000001
sz_imot_percent = 0.001
nms_thres = 0.3
track_length_min = 6 # mon length for Final track
# track_length_min = 25


if (input_mode == "sherbrooke"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
    gt_st_frame = 2754
    gt_ed_frame = 3754
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/sherbrooke_w_l.xml'
        output_file_filtered = '/home/huooi/HL_Results/MOT_result/sherbrooke_w_l_filtered.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/sherbrooke_wo_l.xml'
elif (input_mode == "rouen"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rouen_frames/%08d.jpg'
    gt_st_frame = 20
    gt_ed_frame = 620
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/rouen_w_l.xml'
        output_file_filtered = '/home/huooi/HL_Results/MOT_result/rouen_w_l_filtered.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/rouen_wo_l.xml'
elif (input_mode == "rene"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rene_frames/%08d.jpg'
    gt_st_frame = 7200
    gt_ed_frame = 8199
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/rene_w_l.xml'
        output_file_filtered = '/home/huooi/HL_Results/MOT_result/rene_w_l_filtered.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/rene_wo_l.xml'
elif (input_mode == "stmarc"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/stmarc_frames/%08d.jpg'
    gt_st_frame = 1000
    gt_ed_frame = 1999
    if (ope_mode == "w_label"):
        output_file = '/home/huooi/HL_Results/MOT_result/stmarc_w_l.xml'
        output_file_filtered = '/home/huooi/HL_Results/MOT_result/stmarc_w_l_filtered.xml'
    else:
        output_file = '/home/huooi/HL_Results/MOT_result/stmarc_wo_l.xml'
else:
    print ("no file chosen")


end_frame = gt_ed_frame
# end_frame = 35
# input_file = '/home/huooi/dev/HL_Dataset/UrbanTracker/dum2/%08d.jpg'

# input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
# output_file = '/home/huooi/HL_Results/MOT_xml/xml_sherbrooke'


# input_file = '/usagers/huooi/dev/HL_Dataset/UrbanTracker/dum2/%08d.jpg'
# gt_st_frame = 1
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

def compute_bb_IOU(F1, F2):
    # Reading the list item in Frame [ymin, ymax, xmin, xmax]
    width1 = F1[3] - F1[2]
    height1 = F1[1] - F1[0]

    width2 = F2[3] - F2[2]
    height2 = F2[1] - F2[0]

    start_x = min(F1[2], F2[2])
    end_x = max((F1[2] + width1), (F2[2] + width2))
    width = width1 + width2 - (end_x - start_x)

    start_y = min(F1[0], F2[0])
    end_y = max((F1[0] + height1), (F2[0] + height2))
    height = height1 + height2 - (end_y - start_y)

    if ((width <= 0) or (height <= 0)):
        intersection = 0
        union = (height1 * width1) + (height2 * width2) - (height * width)
        IOU = 0
    else:
        intersection = height * width
        union = (height1 * width1) + (height2 * width2) - (height * width)
        IOU = intersection / float(union)
        # result = (height * width) / float((height1 * width1) + (height2 * width2) - (height * width))
    return [IOU, intersection, union]

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

    # try:
    #     x = int(coord_list[2])
    #     y = int(coord_list[0])
    #     width = int(coord_list[3] - coord_list[2])
    #     height = int(coord_list[1] - coord_list[0])
    # except:
    #     print (coord_list)
    #     print (len(coord_list))
    #     print (coord_list[2])

    return [x, y, width, height]

def reverse_enum(L):
   for index in reversed(xrange(len(L))):
      yield index, L[index]

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_detection(num_detections, scores, detection_thres):
    count = 0
    for i in range((np.squeeze(num_detections))):
        if (scores is None or final_score[i] > detection_thres) and (
                (np.reshape(classes, int(np.squeeze(num_detections)), 1)[i]) != 7.0):
            count = count + 1
            ymin = int(np.round(np.squeeze(boxes)[i, 0] * height))
            xmin = int(np.round(np.squeeze(boxes)[i, 1] * width))
            ymax = int(np.round(np.squeeze(boxes)[i, 2] * height))
            xmax = int(np.round(np.squeeze(boxes)[i, 3] * width))

            crop_img = image_np[ymin:ymax, xmin: xmax]
            center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            nested_hist = np.empty((256, 0), int)
            color = ('b', 'g', 'r')
            for j, col in enumerate(color):
                histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
                histr = histr.astype('float32')
                histr = histr / ((ymax - ymin) * (xmax - xmin))
                nested_hist = np.concatenate((nested_hist, histr), axis=1)
            nested_hist = nested_hist.reshape(256 * 3, 1)
            nested_hist = nested_hist.astype('float32')

            detection_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist,
                                   np.reshape(classes, int(np.squeeze(num_detections)), 1)[i],
                                   final_score[i]])  # for comparison with detection later
            # print ("Box coordinates")
            # print (detection_pack[-1][0])

    return detection_pack

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

viewmode = True
frame_num = 1
skipped_frames = 0

#Tracker Parameters Start
det_score_min = 0.5 #min confidence of detection
cost_feat_thresh = 1.5 # max thres for combined cost (high val for having a stricter matching)
max_frame = 5 # max extended length before track termination for unmatched obj
min_thres_UP_ratio = 0.5 # min UP to whole history ratio
dist_thresh = 50 #distance in terms of pixel
#Tracker Parameters End

trackIdCount = 0
feat_per_frame = []
feat_for_each_frame=[]
track = []
# path =[]
# ls_path =[]
# feat = 2
skip_frame_count = 0

lst = [(255,255,255)]
# track_colors = lst * 1000
track_colors = lst * 1000000

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# tracker = Tracker(90, 10, 8, 3, height, width)
# tracker = Tracker(1.5, 10, 5, 3, height, width, ope_mode)
# tracker = Tracker(1.5, 10, 5, 3, 0.6, height, width, ope_mode)
out = []

tracker = Tracker(cost_feat_thresh, 10, max_frame, 3, dist_thresh, height, width, ope_mode)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # while (True):
    # while (frame_num <629):
    while (frame_num<=end_frame+1):
    # while (cap.isOpened()):
      ret, image_np = cap.read()
      # print ("read status")
      print (ret)
      if (viewmode):
        test_image1 = image_np.copy()
        test_image2 = image_np.copy()
        test_image3 = image_np.copy()
      if not ret:
        # print ("cannot read anymore")
        if out:
            # print ("and there is out")
            # print ("Lee here")
            print (frame_num)
            tracker.alltracks.extend(out)
            # print (len(tracker.alltracks()))
            # for t_rm, trk_rm in enumerate(out):
            #     # tracker.alltracks.append(trk_rm)
            #     print (t_rm)
            #     print ("hello")
            #     tracker.alltracks.extend(trk_rm)
            # tracker.alltracks.append(out)
        # print (type(out))
        # print ("length of out")
        # print (len(out))
        # # print (out)
        # print (len(tracker.alltracks))
        break
      ls_imot = []
      ls_detection = []
      imot_src = '/home/huooi/HL_Dataset/UrbanTracker_output_imot/bgs/' + input_mode + '/' + str(frame_num).zfill(8) + '.png'
      cap_imot = cv2.VideoCapture(imot_src)
      ret_imot, image_imot = cap_imot.read()

      # ret_imot=False
      # print ("imot status")
      # print (imot_src)
      # print (ret_imot)
      if (ret_imot):
        image_imot = cv2.cvtColor(image_imot, cv2.COLOR_BGR2GRAY)
        ret_fg, imot_mask = cv2.threshold(image_imot, 100, 1, cv2.THRESH_BINARY)
        # find contours and get the external one
        image, contours, hier = cv2.findContours(imot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # height, width = image_np.shape[:2]
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) >= (width * height * sz_imot_percent):
              ls_imot.append([y, h + y, x, w + x])


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
          min_score_thresh = det_score_min,
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

      high_detection_thres = 0.7
      low_detection_thres = 0.4
      imot_pack = []
      detection_pack = []
      low_detection_pack = []

      det_input = []
      # all imot to be filtered by detection
      for i in range(len(ls_imot)):
          ymin = ls_imot[i][0]
          xmin = ls_imot[i][2]
          ymax = ls_imot[i][1]
          xmax = ls_imot[i][3]
          crop_img = image_np[ymin:ymax, xmin: xmax]
          center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
          nested_hist = np.empty((256, 0), int)
          color = ('b', 'g', 'r')
          for j, col in enumerate(color):
              histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
              histr = histr.astype('float32')
              histr = histr / ((ymax - ymin) * (xmax - xmin))
              nested_hist = np.concatenate((nested_hist, histr), axis=1)
          nested_hist = nested_hist.reshape(256 * 3, 1)
          nested_hist = nested_hist.astype('float32')
          # hist_per_frame.append(nested_hist)
          # color_per_frame.append(histr)
          # pos_per_frame.append([ymin, ymax, xmin, xmax])  # decide not to normalize position with respect to frame size after all
          # label_per_frame.append([np.reshape(classes, int(np.squeeze(num_detections)), 1)[i], final_score[i]])
          # centers.append(center_pt)

          imot_pack.append([[ymin,ymax,xmin,xmax], center_pt, nested_hist]) #for comparison with detection later

      # low_detection_pack = get_detection(num_detections, scores, low_detection_thres)
      high_detection_pack = get_detection(num_detections, scores, high_detection_thres)
      for i in range(len(high_detection_pack)):
          ymin = high_detection_pack[i][0][0]
          ymax = high_detection_pack[i][0][1]
          xmin = high_detection_pack[i][0][2]
          xmax = high_detection_pack[i][0][3]
          cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255,0,0), 3)

      for i in range(len(imot_pack)):
          ymin = imot_pack[i][0][0]
          ymax = imot_pack[i][0][1]
          xmin = imot_pack[i][0][2]
          xmax = imot_pack[i][0][3]
          cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (0,0,255), 3)

      # cv2.imshow('detection', cv2.resize(test_image2, (int(width), int(height))))
      # cv2.waitKey(0)
      # print (len(high_detection_pack))

      # for aa in range(len(high_detection_pack)):
      #   det_input.append([high_detection_pack[aa][0],
      #                     high_detection_pack[aa][1],
      #                     high_detection_pack[aa][2],
      #                     [high_detection_pack[aa][3], high_detection_pack[aa][4]]
      #                     ])


    # high_detection_thres
      #     #  detection to get label
      # for i in range((np.squeeze(num_detections))):
      #     if (scores is None or final_score[i] > low_detection_thres) and (
      #         (np.reshape(classes, int(np.squeeze(num_detections)), 1)[i]) != 7.0):
      #
      #         ymin = int(np.round(np.squeeze(boxes)[i, 0] * height))
      #         xmin = int(np.round(np.squeeze(boxes)[i, 1] * width))
      #         ymax = int(np.round(np.squeeze(boxes)[i, 2] * height))
      #         xmax = int(np.round(np.squeeze(boxes)[i, 3] * width))
      #
      #         crop_img = image_np[ymin:ymax, xmin: xmax]
      #         center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
      #         nested_hist = np.empty((256, 0), int)
      #         color = ('b', 'g', 'r')
      #         for j, col in enumerate(color):
      #             histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
      #             histr = histr.astype('float32')
      #             histr = histr / ((ymax - ymin) * (xmax - xmin))
      #             nested_hist = np.concatenate((nested_hist, histr), axis=1)
      #         nested_hist = nested_hist.reshape(256 * 3, 1)
      #         nested_hist = nested_hist.astype('float32')
      #
      #         low_detection_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist,
      #                                np.reshape(classes, int(np.squeeze(num_detections)), 1)[i],
      #                                final_score[i]])  # for comparison with detection later

    # #  detection to filter off excessive imot, detection will be taken instead
    #   for i in range((np.squeeze(num_detections))):
    #        if (scores is None or final_score[i] > high_detection_thres) and ((np.reshape(classes, int(np.squeeze(num_detections)), 1)[i]) != 7.0):
    #            count = count + 1
    #
    #            ymin = int (np.round(np.squeeze(boxes)[i, 0] * height))
    #            xmin = int (np.round(np.squeeze(boxes)[i, 1] * width))
    #            ymax = int (np.round(np.squeeze(boxes)[i, 2] * height))
    #            xmax = int (np.round(np.squeeze(boxes)[i, 3] * width))
    #
    #            crop_img = image_np[ymin:ymax, xmin : xmax]
    #            center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
    #            nested_hist = np.empty((256, 0), int)
    #            color = ('b', 'g', 'r')
    #            for j, col in enumerate(color):
    #                histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
    #                histr = histr.astype('float32')
    #                histr = histr / ((ymax - ymin) * (xmax - xmin))
    #                nested_hist = np.concatenate((nested_hist, histr), axis=1)
    #            nested_hist = nested_hist.reshape(256 * 3, 1)
    #            nested_hist = nested_hist.astype('float32')
    #
    #            detection_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist, np.reshape(classes, int(np.squeeze(num_detections)), 1)[i], final_score[i]])  # for comparison with detection later
    #
      # imot_unlabelled =[]
      # imot_rest =[]
      imot_labelled = []
      input_labelled = []
      imot_disregard = []
      detection_labelled = []
      clr_sim_thres  = 0.5
      overlap_sim_thres = 0.05 #0 #0.5
      ######################################################################################
      # pairing might not be one-to-one
      # a metrix of all imot and detection
      pair_matrix = np.zeros(shape=(len(imot_pack),len(high_detection_pack)))

      for i in range(len(imot_pack)):
        for j in range(len(high_detection_pack)):
            clr_cost = cv2.compareHist(imot_pack[i][2], high_detection_pack[j][2], cv2.HISTCMP_BHATTACHARYYA)
            [overlap_cost, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
            # print ("IOU")
            # print (overlap_cost)
            if (overlap_cost > overlap_sim_thres):
                pair_matrix [i,j] = 1

      # print ("Pair matrix here")
      # print (pair_matrix)

      # imot_to_disregard = []
      for j in range(len(high_detection_pack)):
        if ((pair_matrix[:, j].sum()) > 1):
            multi_imot_ind = []
            clr_ls = []
            for i in range(len(imot_pack)):
                if (pair_matrix[i, j] == 1):
                # if (pair_matrix[i, j] == 1 and (pair_matrix[i, :].sum()) == 1): #imot that are simultaneously matched to multiple detection are not studied here
                    multi_imot_ind.append(i)
            # print ("is there multi match here?")
            # print ("testing here")
            # print (pair_matrix[:, j].sum())
            # print (pair_matrix[:, j])
            # print (multi_imot_ind)
            for x, y in itertools.combinations(multi_imot_ind, 2):
                clr_ls.append(cv2.compareHist(imot_pack[x][2], imot_pack[y][2], cv2.HISTCMP_BHATTACHARYYA))
                # print (clr_ls)
        # if (np.var(clr_ls) > ):
        #     if (len(clr_ls)>1):
            #     if (np.variance(clr_ls) >=):
            # else:
            #
            #
            # print ("stats")
            # print (clr_ls)
            # print (np.mean(clr_ls))
            if (np.mean(clr_ls) > clr_sim_thres):
                # print ("take the detection and disregard the imot")
                # print ("check the mean")
                # input_labelled.append (high_detection_pack[j])
                # input_labelled.append([high_detection_pack[j][0], high_detection_pack[j][1], high_detection_pack[j][2], high_detection_pack[j][3]])
                input_labelled.append([high_detection_pack[j][0],
                                       high_detection_pack[j][1],
                                       high_detection_pack[j][2],
                                       [high_detection_pack[j][3], high_detection_pack[j][4]]
                                       ])
                for zz in range (len(multi_imot_ind)):
                    imot_disregard.append (multi_imot_ind[zz])

      for i in range(len(imot_pack)):
        if (i in imot_disregard):
            # print ("Disregard")
            # print (len(imot_disregard))
            continue
        # if (imot_pack[i])
        if ((pair_matrix[i,:].sum())== 0):
            input_labelled.append([imot_pack[i][0], imot_pack[i][1],imot_pack[i][2],[0, 0.50]])
            # print ("Just test from unlabelled imot")
            # print (input_labelled[len(input_labelled) - 1][3])

        elif ((pair_matrix[i,:].sum())== 1):
            det_ind = pair_matrix[i,:].argmax()
            input_labelled.append([imot_pack[i][0], imot_pack[i][1], imot_pack[i][2], [high_detection_pack[det_ind][3], high_detection_pack[det_ind][4]]])
            # print ("Just test from matched imot")
            # print (input_labelled[len(input_labelled) - 1][3])

            #oops can't do this yet
        # elif ((pair_matrix[i,:].sum())> 1):
        else:
            # the case where imot correspond to multiple detection boxes...might happen when there are overlapped detection
            multi_det_ind = []
            multi_det_ind_cost = []
            # ind_multi = np.zeros(shape=(pair_matrix[i, :].sum(), 2))
            for j in range(len(high_detection_pack)):
                if (pair_matrix[i,j] == 1):
                    # multi_itr = 0
                    clr_cost = cv2.compareHist(imot_pack[i][2], high_detection_pack[j][2], cv2.HISTCMP_BHATTACHARYYA)
                    [overlap_cost, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
                    multi_det_ind.append(j)
                    multi_det_ind_cost.append((1-clr_cost) + overlap_cost + high_detection_pack[j][4])
            det_ind = multi_det_ind_cost.index(max(multi_det_ind_cost))
            input_labelled.append([imot_pack[i][0], imot_pack[i][1], imot_pack[i][2],
                                      [high_detection_pack[multi_det_ind[det_ind]][3], high_detection_pack[multi_det_ind[det_ind]][4]]])
            # print ("Just test from imot that matched with multiple detection")
            # print (input_labelled[len(input_labelled) - 1][3])


            # else:
      for i in range(len(input_labelled)):
        ymin = input_labelled[i][0][0]
        ymax = input_labelled[i][0][1]
        xmin = input_labelled[i][0][2]
        xmax = input_labelled[i][0][3]
        cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 255, 255), 2)

      # cv2.imshow('detection', cv2.resize(test_image2, (int(width), int(height))))
      # cv2.waitKey(0)


          # all imot only
      print ("no of total imot")
      imot_labelled = []
      print (len(ls_imot))
      for i in range (len(ls_imot)):
          ymin = ls_imot[i][0]
          xmin = ls_imot[i][2]
          ymax = ls_imot[i][1]
          xmax = ls_imot[i][3]

          ls_detection.append([ymin, ymax, xmin, xmax])
          # print (ymin)
          # print (ls_imot)
          crop_img = image_np[ymin:ymax, xmin: xmax]
          center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]

          # ls_detection.append([ymin, ymax, xmin, xmax])

          nested_hist = np.empty((256, 0), int)
          color = ('b', 'g', 'r')
          for j, col in enumerate(color):
              histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
              histr = histr.astype('float32')
              histr = histr / ((ymax - ymin) * (xmax - xmin))
              nested_hist = np.concatenate((nested_hist, histr), axis=1)
          nested_hist = nested_hist.reshape(256 * 3, 1)
          nested_hist = nested_hist.astype('float32')

          hist_per_frame.append(nested_hist)
          color_per_frame.append(histr)
          pos_per_frame.append([ymin, ymax, xmin, xmax])  # decide not to normalize position with respect to frame size after all
          # label_per_frame.append([np.reshape(classes, int(np.squeeze(num_detections)), 1)[i], final_score[i]])
          centers.append(center_pt)

          # imot_labelled.append([[ymin,ymax,xmin,xmax], center_pt, nested_hist])
          # print ("length hist per frame")
          # print (len(pos_per_frame))
      # for i in range((np.squeeze(num_detections))):
      #   # if scores is None or final_score[i] > 0.5:
      #   if (scores is None or final_score[i] > 0.5) and ((np.reshape(classes,int(np.squeeze(num_detections)),1)[i]) !=7.0):
      #       count = count + 1
      #
      #       ymin = int (np.round(np.squeeze(boxes)[i, 0] * height))
      #       xmin = int (np.round(np.squeeze(boxes)[i, 1] * width))
      #       ymax = int (np.round(np.squeeze(boxes)[i, 2] * height))
      #       xmax = int (np.round(np.squeeze(boxes)[i, 3] * width))
      #       #
      #       # lbl_type = np.reshape(classes, int(np.squeeze(num_detections)), 1)[i_init]
      #       # lbl_conf = final_score[i_init] * 100
      #       # box_nms = [xmin_old, xmax_old, ymin_old, ymax_old, lbl_type, lbl_conf]
      #       # box_nms_block.append(box_nms)
      #
      #       crop_img = image_np[ymin:ymax, xmin: xmax]
      #       center_pt = [(xmin+xmax)/2, (ymin+ymax)/2]
      #
      #       ls_detection.append([ymin, ymax, xmin, xmax])
      #
      #       nested_hist = np.empty((256, 0), int)
      #       color = ('b', 'g', 'r')
      #       for j, col in enumerate(color):
      #           histr = cv2.calcHist([crop_img], [j], None, [256], [0, 256])
      #           histr = histr.astype('float32')
      #           histr = histr / ((ymax - ymin) * (xmax - xmin))
      #           nested_hist = np.concatenate((nested_hist, histr), axis=1)
      #       nested_hist = nested_hist.reshape(256*3, 1)
      #       nested_hist = nested_hist.astype('float32')
      #
      #       hist_per_frame.append(nested_hist)
      #       color_per_frame.append(histr)
      #       pos_per_frame.append([ymin, ymax, xmin, xmax]) # decide not to normalize position with respect to frame size after all
      #       label_per_frame.append([np.reshape(classes,int(np.squeeze(num_detections)),1)[i], final_score[i]])
      #
      #       centers.append(center_pt)
      #
      #
      #
      # # imot_selected = []
      # # for i_imot in ls_imot:
      # #     for i_det in ls_detection:
      # # #   for i_imot in ls_imot:
      # #         [iou, itrsect, union] = compute_bb_IOU(i_imot, i_det)
      # #         if (iou > 0):
      # #             imot_selected.append(i_imot)
      # #             break
      #
      # if (ret_imot):
      #     imot_selected = []
      #     for i_imot in ls_imot:
      #         for i_det in ls_detection:
      #             [iou, itrsect, union] = compute_bb_IOU(i_imot, i_det)
      #             if (iou > 0):
      #                 imot_selected.append(i_imot)
      #                 break
      #
      #     imot = [item for item in ls_imot if item not in imot_selected]
      #     # unlabelled imot that do not coincide with detection
      #     for inew_imot in imot:
      #         crop_imot = image_np[inew_imot[0]:inew_imot[1], inew_imot[2]: inew_imot[3]]
      #         center_pt = [(inew_imot[2] + inew_imot[3]) / 2, (inew_imot[0] + inew_imot[1]) / 2]
      #         nested_hist = np.empty((256, 0), int)
      #         color = ('b', 'g', 'r')
      #
      #         for j, col in enumerate(color):
      #             histr = cv2.calcHist([crop_imot], [j], None, [256], [0, 256])
      #             histr = histr.astype('float32')
      #             histr = histr / ((inew_imot[1] - inew_imot[0]) * (inew_imot[3] - inew_imot[2]))
      #             nested_hist = np.concatenate((nested_hist, histr), axis=1)
      #         nested_hist = nested_hist.reshape(256 * 3, 1)
      #         nested_hist = nested_hist.astype('float32')
      #
      #         hist_per_frame.append(nested_hist)
      #         color_per_frame.append(histr)
      #         pos_per_frame.append([inew_imot[0], inew_imot[1], inew_imot[2], inew_imot[3]])  # decide not to normalize position with respect to frame size after all
      #         label_per_frame.append ([0, 0.50])
      #         centers.append(center_pt)
      #         # label_per_frame.append([np.reshape(classes, int(np.squeeze(num_detections)), 1)[i], final_score[i]])
      #
      #         # cv2.imshow("imot", crop_imot)
      #         # cv2.waitKey(0)
      #         # cv2.destroyAllWindows()


          #check exactly the items in ls_imot, imot_selected, and imot
          # if (viewmode):
              # for each_item in ls_imot:
              #   clr = 2
              #   color = track_colors[clr]
              #   cv2.drawMarker(test_image1, (int(each_item[2]), int(each_item[1])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image1, (int(each_item[3]), int(each_item[1])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image1, (int(each_item[2]), int(each_item[0])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image1, (int(each_item[3]), int(each_item[0])), color, 3, 20, 1, 8)
              #   cv2.rectangle(test_image1, (int(each_item[2]), int(each_item[1])), (int(each_item[3]), int(each_item[0])),
              #               color, 2)
              #
              # for each_item in imot_selected:
              #   clr = 3
              #   color = track_colors[clr]
              #   cv2.drawMarker(test_image2, (int(each_item[2]), int(each_item[1])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image2, (int(each_item[3]), int(each_item[1])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image2, (int(each_item[2]), int(each_item[0])), color, 3, 20, 1, 8)
              #   cv2.drawMarker(test_image2, (int(each_item[3]), int(each_item[0])), color, 3, 20, 1, 8)
              #   cv2.rectangle(test_image2, (int(each_item[2]), int(each_item[1])), (int(each_item[3]), int(each_item[0])),
              #           color, 2)
              #
              # for each_item in imot:
              #   clr = 4
              #   color = track_colors[clr]
              # # cv2.drawMarker(test_image3, (int(each_item[2]), int(each_item[1])), color, 3, 20, 1, 8)
              # # cv2.drawMarker(test_image3, (int(each_item[3]), int(each_item[1])), color, 3, 20, 1, 8)
              # # cv2.drawMarker(test_image3, (int(each_item[2]), int(each_item[0])), color, 3, 20, 1, 8)
              # # cv2.drawMarker(test_image3, (int(each_item[3]), int(each_item[0])), color, 3, 20, 1, 8)
              # #   cv2.rectangle(test_image3, (int(each_item[2]), int(each_item[1])), (int(each_item[3]), int(each_item[0])), color, 2)
              #   cv2.rectangle(test_image3, (int(each_item[2]), int(each_item[1])), (int(each_item[3]), int(each_item[0])),
              #               color, 2)
              #
              # cv2.imshow('ls_imot', cv2.resize(test_image1, (int(width), int(height))))
              # cv2.imshow('imot_selected', cv2.resize(test_image2, (int(width), int(height))))
              # cv2.imshow('imot', cv2.resize(test_image3, (int(width), int(height))))
              # cv2.imshow('imot', cv2.resize(test_image3, (int(width), int(height))))
              # cv2.imshow('detection', cv2.resize(image_np, (int(width), int(height))))
              # cv2.waitKey(0)

    # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      print("Frame " + str(frame_num) + " with "+ str(count) + " objects")
      print (len(pos_per_frame))


      # if (ret_imot):
      #   print ("number of imot")
      #   print (len(imot))
      print ("number of detection")
      print (len(ls_detection))

      feat_per_frame = [pos_per_frame, centers, hist_per_frame] #imot only

      # feat_per_frame = [pos_per_frame, centers, hist_per_frame, label_per_frame]

      # 0 is pos_iou, 1 is pos_bb, 2 is pos_c, 3 is color, 4 is label
      feature_index = [1,3,4]
      feature_index = [1, 3]
      # out = tracker.UpdateTracker(feat_per_frame, feature_index, frame_num)

      # HL 2019 change the why feature are fed into the tracker
      out = tracker.UpdateTracker(imot_pack, feature_index, frame_num)

      # out = tracker.UpdateTracker(input_labelled, feature_index, frame_num)
      # print ("checking detection")
      # print (det_input[len(det_input) - 1][0])
      # print (det_input[len(det_input) - 1][1])
      # print (det_input[len(det_input) - 1][3])
      # print (det_input[len(det_input) - 1][3])
      # out = tracker.UpdateTracker(det_input, feature_index, frame_num)
      # print ("length of out")
      # print (len(out))
      # print (out)


      # for t, trk in enumerate((out)):
      #   clr = (trk.retrieve_id())
      #   color = track_colors[clr]
      #   output_bb = trk.retrieve_state()
      #   cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[1])), color, 3, 20, 1, 8)
      #   cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[1])), color, 3, 20, 1, 8)
      #   cv2.drawMarker(image_np, (int(output_bb[2]), int(output_bb[0])), color, 3, 20, 1, 8)
      #   cv2.drawMarker(image_np, (int(output_bb[3]), int(output_bb[0])), color, 3, 20, 1, 8)

      # if not(frame_num == 0):
      # if (frame_num >= gt_st_frame):
      # if ((frame_num >= gt_st_frame) and (frame_num <= gt_ed_frame)):
      #     if not (os.path.isfile(output_file)):
      #         # print ("no file")
      #         video = Element('Video')
      #         video.attrib['fname'] = input_file  # must be str; cannot be an int
      #         video.attrib['start_frame'] = str(frame_num)
      #         video.attrib['end_frame'] = str(frame_num)
      #
      #         for t, trk in enumerate((out)):
      #             output_bb = trk.retrieve_state()
      #
      #             traj = Element("Trajectory", obj_id=str(trk.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk.retrieve_type())[0], start_frame=str(frame_num), end_frame=str(frame_num))
      #             # SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
      #             SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[1]- output_bb[0]),
      #                        width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]), frame_no=str(frame_num))
      #             video.append(traj)
      #         # open(output_file, "w").write(prettify(video))
      #     else:
      #         doc = ET.parse(output_file)
      #         root = doc.getroot()
      #         for t, trk in enumerate((out)):
      #             output_bb = trk.retrieve_state()
      #
      #             if (root.find(".//Trajectory[@obj_id='" + str(trk.retrieve_id()) + "']")) is None:
      #                 # create traj element and frame subelement, update root attribute
      #                 # print ("cannot find, gotta create new element")
      #                 traj = Element("Trajectory", obj_id=str(trk.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk.retrieve_type())[0], start_frame=str(frame_num), end_frame=str(frame_num))
      #                 # SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
      #                 SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[1] - output_bb[0]), width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]),
      #                            frame_no=str(frame_num))
      #                 root.append(traj)
      #             else:
      #                 # create frame subelement on the correct traj element, update root and element attribute
      #                 # print ("there is such element")
      #                 for node in root.findall(".//Trajectory[@obj_id='" + str(trk.retrieve_id()) + "']"):
      #                     node.set('end_frame', "'" + str(frame_num) + "'")
      #                     # SubElement(node, 'Frame', contour_pt="0", annotation="0", observation="0", height=str(output_bb[3]), width=str(output_bb[2]), y=str(output_bb[1]), x=str(output_bb[0]), frame_no=str(frame_num))
      #                     SubElement(node, 'Frame', contour_pt="0", annotation="0", observation="0",
      #                                height=str(output_bb[1] - output_bb[0]), width=str(output_bb[3] - output_bb[2]), y=str(output_bb[0]), x=str(output_bb[2]), frame_no=str(frame_num))
      #
      #         root.set('end_frame', "'" + str(frame_num) + "'")
      #
      #         for elem in root.iter('*'):
      #             if elem.text is not None:
      #                 elem.text = elem.text.strip()
      #             if elem.tail is not None:
      #                 elem.tail = elem.tail.strip()
      #
      #         # Now we have a new well-formed XML document. It is not very nicely formatted...
      #         xml_out = ET.tostring(root)
      #         # ...so we'll use minidom to make the output a little prettier
      #         dom = minidom.parseString(xml_out)
      #         # print dom.toprettyxml()
      #         # open(output_file, "w").write(dom.toprettyxml())


      # if ((frame_num >= gt_st_frame) and (frame_num<= gt_ed_frame)):
      #   cv2.imwrite("/home/huooi/HL_Results/MOT_result/REDO_exp/frame_rouen_w_l/both%d.jpg" % frame_num,
      #           cv2.resize(image_np, (int(width), int(height))))  # saving frames

      # cv2.imshow('object detection', cv2.resize(image_np, (int(width), int(height))))
      # cv2.waitKey(3)

      frame_num = frame_num + 1

    # video = Element('Video')
    # video.attrib['fname'] = input_file  # must be str; cannot be an int
    # video.attrib['start_frame'] = str(gt_st_frame)
    # video.attrib['end_frame'] = str(gt_ed_frame)
    #
    # # Original version: write all
    # for t_all, trk_all in enumerate(tracker.alltracks):
    #     # trk_start_fr = trk_all.retrieve_frame_start()
    #     # trk_end_fr = trk_all.retrieve_frame_end()
    #
    #     length_hist = len(trk_all.retrieve_hist_whole())
    #     trk_start_fr = trk_all.retrieve_time_stamp_all()[0][0]
    #     trk_end_fr = trk_all.retrieve_time_stamp_all()[length_hist - 1][0]
    #
    #     # traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk_all.retrieve_type())[0], start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
    #     traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()),
    #                    obj_type="",
    #                    start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
    #     for t_info, trk_info in enumerate(trk_all.retrieve_hist_whole()):
    #         SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0",
    #                height=str(trk_all.retrieve_hist(t_info)[1] - trk_all.retrieve_hist(t_info)[0]), width=str(trk_all.retrieve_hist(t_info)[3] - trk_all.retrieve_hist(t_info)[2]), y=str(trk_all.retrieve_hist(t_info)[0]), x=str(trk_all.retrieve_hist(t_info)[2]), frame_no=str(trk_all.retrieve_time_stamp(t_info)[0]))
    #     video.append(traj)
    #
    # open(output_file, "w").write(prettify(video))
    # print (len(tracker.alltracks))
    #
    # print ("cest  fini")

    video = Element('Video')
    video.attrib['fname'] = input_file  # must be str; cannot be an int
    video.attrib['start_frame'] = str(gt_st_frame)
    video.attrib['end_frame'] = str(gt_ed_frame)

    print ("why")
    print (len(tracker.alltracks))
    # Filtered version:
    i_n_element =0
    i_ratio = 0
    i_UP = 0
    for i, trk_all in reverse_enum(tracker.alltracks):
        # trk_start_fr = trk_all.retrieve_frame_start()
        # trk_end_fr = trk_all.retrieve_frame_end()
        length_hist = len(trk_all.retrieve_hist_whole())

        # Remove track that is shorter than minimum length
        if (length_hist < track_length_min):
            tracker.alltracks.pop(i)
            # i -= 1
            i_n_element = i_n_element + 1
            print ("this one is removed due to length")
            # print (i)
            print (trk_all.retrieve_id())
            continue


        trk_start_fr = trk_all.retrieve_time_stamp_all()[0][0]
        trk_end_fr = trk_all.retrieve_time_stamp_all()[length_hist-1][0]

        # print ("test new function")
        # print (trk_all.retrieve_frame_start())  #frame when the obj start appear #when calling the list dont use this, use ..[0]
        # print (trk_all.retrieve_frame_end())   #frame when the obj apper last

        # print (len(trk_all.retrieve_time_stamp_all()))
        # print (len(trk_all.retrieve_hist_all()))

        # if ((len(trk_all.retrieve_time_stamp_all()))!=(len(trk_all.retrieve_hist_all()))):
        #     print ("oops!!! not the same length")

        print (trk_all.retrieve_time_stamp_all())

        hist_frame_first = trk_all.retrieve_hist(0)
        # sz = len(trk_all.retrieve_time_stamp_all())
        hist_frame_last = trk_all.retrieve_hist(length_hist-1)
        # print (trk_all.retrieve_hist_all())

        # delta_x = abs(trk_all.retrieve_hist(trk_end_fr)[2] - trk_all.retrieve_hist(trk_start_fr)[2])
        # delta_y = abs(trk_all.retrieve_hist(trk_end_fr)[0] - trk_all.retrieve_hist(trk_start_fr)[0])

        delta_x = abs(hist_frame_last[2] - hist_frame_first[2])
        delta_y = abs(hist_frame_last[0] - hist_frame_first[0])
        ratio_x = delta_x / length_hist
        ratio_y = delta_y / length_hist

        # if ((trk_all.retrieve_id() == 29) or (trk_all.retrieve_id() == 50) or (trk_all.retrieve_id() == 58)):
        #     print ("testing specific cases now: ")
        #     print (trk_all.retrieve_id())
        #     print  (delta_x)
        #     print  (delta_y)
        #     print  (ratio_x)
        #     print  (ratio_y)

        # # if (ratio_x <= 0.5 and ratio_y <= 0.5):
        # if (ratio_x <= 0.5 or ratio_y <= 0.5):
        #     tracker.alltracks.pop(i)
        #     # i -= 1
        #     i_ratio = i_ratio + 1
        #     print ("this one is removed due to ratio")
        #     # print (i)
        #     print (trk_all.retrieve_id())
        #     continue

        # if (last_y == -1 and last_x == -1):
        #     tracker.alltracks.pop(i)
        #     continue

        # Remove track that is only composed mostly of UP (unreliable prediction)
        iter_UP = 0
        for i_timestamp in trk_all.retrieve_time_stamp_all():
            if (i_timestamp[1] == "UP"):
                iter_UP = iter_UP + 1

        if ((iter_UP / (len(trk_all.retrieve_time_stamp_all()))) >= min_thres_UP_ratio):
            tracker.alltracks.pop(i)
            i_UP = i_UP + 1
            print ("this one is removed due to UP")
            print (trk_all.retrieve_id())
            continue

        # traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk_all.retrieve_type())[0], start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
        traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()),
                       obj_type="",
                       start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
        i -= 1

        for t_info, trk_info in enumerate(trk_all.retrieve_hist_whole()):
            SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0",
                   height=str(trk_all.retrieve_hist(t_info)[1] - trk_all.retrieve_hist(t_info)[0]), width=str(trk_all.retrieve_hist(t_info)[3] - trk_all.retrieve_hist(t_info)[2]), y=str(trk_all.retrieve_hist(t_info)[0]), x=str(trk_all.retrieve_hist(t_info)[2]), frame_no=str(trk_all.retrieve_time_stamp(t_info)[0]))
        video.append(traj)

        open(output_file_filtered, "w").write(prettify(video))
    print (len(tracker.alltracks))

    print ("cest  fini encore")
    print (i_n_element)
    print (i_ratio)
    print (i_UP)
