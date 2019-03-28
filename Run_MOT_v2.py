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
from xml.etree import ElementTree as ET
import cv2

# ## Object detection imports
# imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

input_mode = "rouen" #"sherbrooke","rouen","rene","stmarc"
ope_mode = "w_label" #"w_label", "wo_label"
# sz_imot_percent = 0.0000001
sz_imot_percent = 0.001 # 0.001 for the rest except rene (0.00001,  0.001, 0.0000001 )
nms_thres = 0.3
track_length_min = 6 # min length for Final track 6
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
try:
    os.remove(output_file)
except OSError:
    pass

cap = cv2.VideoCapture(input_file)


# MIO finetuned weight from COCO
PATH_TO_CKPT = '/home/huooi/HL_Proj/PycharmProjects/models-master/research/object_detection/HL_rfcn_resnet101_mio/frozen/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/huooi/HL_Proj/PycharmProjects/models/object_detection/data/mio_label_map.pbtxt'
NUM_CLASSES = 11

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

def reverse_enum(L):
   for index in reversed(xrange(len(L))):
      yield index, L[index]

def prettify(elem):
    #Return a pretty-printed XML string for the Element.
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_detection(num_detections, scores, detection_thres, det_class, box, w, h, img):
    count = 0
    det_pack = []
    all_scores = np.squeeze(scores)
    for i in range((np.squeeze(num_detections))):
        if (scores is None or all_scores[i] > detection_thres) and (
                (np.reshape(det_class, int(np.squeeze(num_detections)), 1)[i]) != 7.0):
            count = count + 1
            ymin = int(np.round(np.squeeze(box)[i, 0] * h))
            xmin = int(np.round(np.squeeze(box)[i, 1] * w))
            ymax = int(np.round(np.squeeze(box)[i, 2] * h))
            xmax = int(np.round(np.squeeze(box)[i, 3] * w))

            crop_img = img[ymin:ymax, xmin: xmax]
            center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
            # nested_hist = np.empty((256, 0), int)
            nested_hist = np.empty((clr_bin_count, 0), int)
            color = ('b', 'g', 'r')
            for j, col in enumerate(color):
                histr = cv2.calcHist([crop_img], [j], None, [clr_bin_count], [0, 256])
                histr = histr.astype('float32')
                histr = histr / ((ymax - ymin) * (xmax - xmin))
                nested_hist = np.concatenate((nested_hist, histr), axis=1)
            # nested_hist = nested_hist.reshape(256 * 3, 1)
            nested_hist = nested_hist.reshape(clr_bin_count * 3, 1)
            nested_hist = nested_hist.astype('float32')

            # det_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist,
            #                        np.reshape(det_class, int(np.squeeze(num_detections)), 1)[i],
            #                        all_scores[i]])  # for comparison with detection later
            det_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist,
                             [np.reshape(det_class, int(np.squeeze(num_detections)), 1)[i], all_scores[i]]])  # for comparison with detection later


    return det_pack

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#
# Tracker Setting Start
viewmode = True
frame_num = 1
skipped_frames = 0
# Tracker Setting End

# Tracker Parameters Start
det_score_min = 0.5 #min confidence of detection
cost_feat_thresh = 1.5 # max thres for combined cost (high val for having a more lenient matching)
max_frame = 5 # max extended length before track termination for unmatched obj
min_thres_UP_ratio = 0.5 # min UP to whole history ratio
min_thres_BP_ratio = 0.6 # min BP to whole history ratio
dist_thresh = 70 #distance in terms of pixel 90, 70, 50, 30, 10
clr_bin_count = 256 #256, 64, 32, 16,9, 4
trj_step_overlap_thres =0.01  #0.5, 0.1, 0.05, 0.03, 0.01
# Tracker Parameters End

trackIdCount = 0
feat_per_frame = []
feat_for_each_frame=[]
track = []
skip_frame_count = 0

lst = [(255,255,255)]
track_colors = lst * 1000000

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

out = []
tracker = Tracker(cost_feat_thresh, max_frame, trj_step_overlap_thres ,3, dist_thresh, height, width, ope_mode)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while (True):
    # while (frame_num<=end_frame+1):
    # if (frame_num <= end_frame + 1):
      ret, image_np = cap.read()
      if (not ret or (frame_num > end_frame + 1)):
      # if not ret:
          print ("print this when it happen")
          tracker.alltracks.extend(out) # just to combine the existing tracks from the previous frames with the accumulated tracks that have been terminated previously
                # should only run at the very last frame to produce final trajectories
          break
      if (viewmode):
        test_image1 = image_np.copy()
        test_image2 = image_np.copy()
        test_image3 = image_np.copy()
      # if not ret:
      #   if out:
      #       print ("print this when it happen")
      #       tracker.alltracks.extend(out) # just to combine the existing tracks from the previous frames with the accumulated tracks that have been terminated previously
      #       # should only run at the very last frame to produce final trajectories
      #   break
      ls_imot = []
      ls_detection = []
      imot_src = '/home/huooi/HL_Dataset/UrbanTracker_output_imot/bgs/' + input_mode + '/' + str(frame_num).zfill(8) + '.png'
      cap_imot = cv2.VideoCapture(imot_src)
      ret_imot, image_imot = cap_imot.read()

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

      # cv2.imshow('object detection', cv2.resize(image_np, (int(width), int(height))))
      # cv2.waitKey(0)

      # count = 0
      # box_nms_block = []
      # centers = []
      # color_per_frame = []
      # pos_per_frame = []
      # hist_per_frame=[]
      # label_per_frame=[]

      high_detection_thres = 0.4
      # low_detection_thres = 0.4
      imot_pack = []

      # all imot to be filtered by detection
      for i in range(len(ls_imot)):
          ymin = ls_imot[i][0]
          xmin = ls_imot[i][2]
          ymax = ls_imot[i][1]
          xmax = ls_imot[i][3]
          crop_img = image_np[ymin:ymax, xmin: xmax]
          center_pt = [(xmin + xmax) / 2, (ymin + ymax) / 2]
          # nested_hist = np.empty((256, 0), int)
          nested_hist = np.empty((clr_bin_count, 0), int)
          color = ('b', 'g', 'r')
          for j, col in enumerate(color):
              histr = cv2.calcHist([crop_img], [j], None, [clr_bin_count], [0, 256])
              histr = histr.astype('float32')
              histr = histr / ((ymax - ymin) * (xmax - xmin))
              nested_hist = np.concatenate((nested_hist, histr), axis=1)
          nested_hist = nested_hist.reshape(clr_bin_count * 3, 1)
          # nested_hist = nested_hist.reshape(256 * 3, 1)
          nested_hist = nested_hist.astype('float32')
          imot_pack.append([[ymin, ymax, xmin, xmax], center_pt, nested_hist, [0, 0.50]])  # for comparison with detection later

      high_detection_pack = get_detection(num_detections, scores, high_detection_thres, classes, boxes, width, height, image_np)

      for i in range(len(high_detection_pack)):
        ymin = high_detection_pack[i][0][0]
        ymax = high_detection_pack[i][0][1]
        xmin = high_detection_pack[i][0][2]
        xmax = high_detection_pack[i][0][3]
        cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 3)

      for i in range(len(imot_pack)):
        ymin = imot_pack[i][0][0]
        ymax = imot_pack[i][0][1]
        xmin = imot_pack[i][0][2]
        xmax = imot_pack[i][0][3]
        cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (0, 0, 255), 3)

      ######################################################################################
      # Matching IM
      ######################################################################################
      imot_labelled = []
      input_labelled = []
      imot_disregard = []
      detection_labelled = []
      clr_sim_thres = 0.5
      overlap_sim_thres = 0.05  # 0 #0.5
      overlap_merge_thres = 0.5  # 0 #0.5
      ######################################################################################
      pair_matrix = np.zeros(shape=(len(imot_pack), len(high_detection_pack)))

      for i in range(len(imot_pack)):
        for j in range(len(high_detection_pack)):
            clr_cost = cv2.compareHist(imot_pack[i][2], high_detection_pack[j][2], cv2.HISTCMP_BHATTACHARYYA)
            [overlap_input, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
            if (overlap_input > overlap_sim_thres):
                pair_matrix[i, j] = 1

      for j in range(len(high_detection_pack)):
        if ((pair_matrix[:, j].sum()) > 1):
            multi_imot_ind = []
            clr_ls = []
            for i in range(len(imot_pack)):
                # [overlap_cost, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
                if (pair_matrix[i, j] == 1):
                    [overlap_box, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
                    if (overlap_box > overlap_merge_thres):
                        multi_imot_ind.append(i)
                    # multi_imot_ind.append(i)
            for x, y in itertools.combinations(multi_imot_ind, 2):
                clr_ls.append(cv2.compareHist(imot_pack[x][2], imot_pack[y][2], cv2.HISTCMP_BHATTACHARYYA))
            # if (np.mean(clr_ls) > clr_sim_thres):
            if (np.mean(clr_ls) < clr_sim_thres):
                input_labelled.append([high_detection_pack[j][0],
                                       high_detection_pack[j][1],
                                       high_detection_pack[j][2],
                                       high_detection_pack[j][3]
                                       ])
                for zz in range(len(multi_imot_ind)):
                    imot_disregard.append(multi_imot_ind[zz])


      for i in range(len(imot_pack)):
        if (i in imot_disregard):
            continue
        if ((pair_matrix[i,:].sum())== 0):
            input_labelled.append([imot_pack[i][0], imot_pack[i][1],imot_pack[i][2],[0, 0.50]])
        elif ((pair_matrix[i,:].sum())== 1):
            det_ind = pair_matrix[i,:].argmax()
            input_labelled.append([imot_pack[i][0], imot_pack[i][1], imot_pack[i][2], high_detection_pack[det_ind][3]])
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
                    [overlap_input, temp, temp] = compute_bb_IOU(imot_pack[i][0], high_detection_pack[j][0])
                    multi_det_ind.append(j)
                    # multi_det_ind_cost.append((1-clr_cost) + overlap_cost + high_detection_pack[j][3][1])
                    multi_det_ind_cost.append((1 - clr_cost) + overlap_input)
            det_ind = multi_det_ind_cost.index(max(multi_det_ind_cost))
            input_labelled.append([imot_pack[i][0], imot_pack[i][1], imot_pack[i][2],
                                      high_detection_pack[multi_det_ind[det_ind]][3]])


      for i in range(len(input_labelled)):
        ymin = input_labelled[i][0][0]
        ymax = input_labelled[i][0][1]
        xmin = input_labelled[i][0][2]
        xmax = input_labelled[i][0][3]
        cv2.rectangle(test_image2, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 255, 255), 2)

      cv2.imshow('preview', cv2.resize(test_image2, (int(width), int(height))))
      cv2.waitKey(0)
      cv2.imwrite("/home/huooi/HL_Results/MOT_result/fig/%d.jpg" % frame_num, test_image2)

      print ("Frame " + str (frame_num))
      print ("number of imot")
      print (len(ls_imot))

      print ("number of detection")
      print (len(high_detection_pack))

      # 0 is pos_iou, 1 is pos_bb, 2 is pos_c, 3 is color, 4 is label
      feature_index = [1,3,4]
      # feature_index = [4]
      # out = tracker.UpdateTracker(high_detection_pack, feature_index, frame_num)
      # feature_index = [1, 3]
      # out = tracker.UpdateTracker(imot_pack, feature_index, frame_num)
      out = tracker.UpdateTracker(input_labelled, feature_index, frame_num)

      print ("number of track output for this frame ")
      print (len(out))
      print (len(tracker.alltracks))

      frame_num = frame_num + 1

    video = Element('Video')
    video.attrib['fname'] = input_file  # must be str; cannot be an int
    video.attrib['start_frame'] = str(gt_st_frame)
    video.attrib['end_frame'] = str(gt_ed_frame)

    # Filtered version:
    i_n_element = 0
    i_ratio = 0
    i_UP = 0
    for i, trk_all in reverse_enum(tracker.alltracks):
        length_hist = len(trk_all.retrieve_hist_whole())

        # Remove track that is shorter than minimum length
        if (length_hist < track_length_min):
            tracker.alltracks.pop(i)
            i_n_element = i_n_element + 1
            print ("this one is removed due to length")
            print (trk_all.retrieve_id())
            continue

        trk_start_fr = trk_all.retrieve_time_stamp_all()[0][0]
        trk_end_fr = trk_all.retrieve_time_stamp_all()[length_hist - 1][0]

        # print (trk_all.retrieve_time_stamp_all())

        # hist_frame_first = trk_all.retrieve_hist(0)
        # hist_frame_last = trk_all.retrieve_hist(length_hist - 1)
        #
        # delta_x = abs(hist_frame_last[2] - hist_frame_first[2])
        # delta_y = abs(hist_frame_last[0] - hist_frame_first[0])
        # ratio_x = delta_x / length_hist
        # ratio_y = delta_y / length_hist

        # if (ratio_x <= 0.05 and ratio_y <= 0.05):
        #   # if (ratio_x <= 0.5 or ratio_y <= 0.5):
        #     tracker.alltracks.pop(i)
        #     # i -= 1
        #     i_ratio = i_ratio + 1
        #     print ("this one is removed due to ratio")
        #     # print (i)
        #     print (trk_all.retrieve_id())
        #     continue


        # Remove track that is only composed mostly of UP (unreliable prediction)
        iter_UP = 0
        iter_BP = 0
        for i_timestamp in trk_all.retrieve_time_stamp_all():
            if (i_timestamp[1] == "UP"):
                iter_UP = iter_UP + 1
            if (i_timestamp[1] == "BP"):
                iter_BP = iter_BP + 1

        if ((iter_UP / (len(trk_all.retrieve_time_stamp_all()))) >= min_thres_UP_ratio):
            tracker.alltracks.pop(i)
            i_UP = i_UP + 1
            print ("this one is removed due to UP")
            print (trk_all.retrieve_id())
            continue

        if ((iter_BP / (len(trk_all.retrieve_time_stamp_all()))) >= min_thres_BP_ratio):
            tracker.alltracks.pop(i)
            i_BP = i_BP + 1
            print ("this one is removed due to BP")
            print (trk_all.retrieve_id())
            continue

        traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()), obj_type=convert_type_to_urbantracker_format(trk_all.retrieve_type())[0], start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
        # traj = Element("Trajectory", obj_id=str(trk_all.retrieve_id()),
        #                  obj_type="",
        #                  start_frame=str(trk_start_fr), end_frame=str(trk_end_fr))
        i -= 1

        for t_info, trk_info in enumerate(trk_all.retrieve_hist_whole()):
            SubElement(traj, 'Frame', contour_pt="0", annotation="0", observation="0",
                         height=str(trk_all.retrieve_hist(t_info)[1] - trk_all.retrieve_hist(t_info)[0]),
                         width=str(trk_all.retrieve_hist(t_info)[3] - trk_all.retrieve_hist(t_info)[2]),
                         y=str(trk_all.retrieve_hist(t_info)[0]), x=str(trk_all.retrieve_hist(t_info)[2]),
                         frame_no=str(trk_all.retrieve_time_stamp(t_info)[0]))
        video.append(traj)
        open(output_file_filtered, "w").write(prettify(video))

    print (len(tracker.alltracks))

    print ("Finished")
    print (i_n_element)
    print (i_ratio)
    print (i_UP)



