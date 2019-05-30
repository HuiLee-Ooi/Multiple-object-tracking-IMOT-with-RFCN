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

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# MIO finetuned weight from COCO
# PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/MIO/mio-ssd-inc/train/output_inference_graph3.pb'
# # PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/MIO/faster_rcnn_101/train/output_inference_graph2.pb'
# PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_rfcn_resnet101_mio/train/output_inference_graph.pb'
PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_rfcn_resnet101_mio/frozen/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mio_label_map.pbtxt'
NUM_CLASSES = 11

input_mode = "stmarc" #"stmarc","rouen","rene","stmarc"
detection_thres = 0.5

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


output_folder = '/home/huooi/HL_Results/MOT_detresult/'
if (input_mode == "sherbrooke"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/sherbrooke_frames/%08d.jpg'
    output_file = output_folder + 'sherbrooke.txt'
elif (input_mode == "rouen"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rouen_frames/%08d.jpg'
    output_file = output_folder + 'rouen.txt'
elif (input_mode == "rene"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/rene_frames/%08d.jpg'
    output_file = output_folder + 'rene.txt'
elif (input_mode == "stmarc"):
    input_file = '/home/huooi/HL_Dataset/UrbanTracker/stmarc_frames/%08d.jpg'
    output_file = output_folder + 'stmarc.txt'
else:
    print ("no file chosen")
cap = cv2.VideoCapture(input_file)

frame_num = 1
outF = open(output_file, "w")
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
    # while (cap.isOpened()):
      ret, image_np = cap.read()
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
          min_score_thresh=.5,
          use_normalized_coordinates=True,
          line_thickness=2,
      )
      final_score = np.squeeze(scores)
      count = 0
      for i in range((np.squeeze(num_detections))):
          if (scores is None or final_score[i] > detection_thres) and ((np.reshape(classes, int(np.squeeze(num_detections)), 1)[i]) != 7.0):
              count = count + 1
              ymin = int(np.round(np.squeeze(boxes)[i, 0] * height))
              xmin = int(np.round(np.squeeze(boxes)[i, 1] * width))
              ymax = int(np.round(np.squeeze(boxes)[i, 2] * height))
              xmax = int(np.round(np.squeeze(boxes)[i, 3] * width))
              outF.write(str(frame_num) + "," + str(ymin) + "," + str(ymax) + "," + str(xmin) + "," + str(xmax) + "," + str(np.reshape(classes, int(np.squeeze(num_detections)), 1)[i]) + "," + str(final_score[i]))
              outF.write("\n")
      print("Frame " + str(frame_num) + " with " + str(count) + " objects")
      # cv2.imshow('object detection', cv2.resize(image_np, (width, height)))
      # cv2.waitKey(5)
      frame_num = frame_num + 1
    outF.close()
