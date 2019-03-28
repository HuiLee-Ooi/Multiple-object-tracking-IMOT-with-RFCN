# Import python libraries
import numpy as np
from kalmanfilter2track import KalmanFilterObj
from filterpy.kalman import KalmanFilter
# from common import dprint
from scipy.optimize import linear_sum_assignment
import cv2

class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, min_history, im_height, im_width):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            min_history: minimum number of detection to disregard false detection
            trackCount: count of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.min_history = min_history
        # self.hits = 0
        self.tracks_per_frame = []
        self.alltracks = []
        self.active_tracks = []
        self.im_height = im_height
        self.im_width = im_width
        self.trackCount = 0
        self.framecount = 0

    def FeatureExtractorbyFeat(self, list_in, feat_index):
        # Extract and Reshape list[feat][obj] according to called feature index
        list_ori = list_in[feat_index]
        list_selected = []
        for i in range(len(list_ori)):
            list_selected.append(np.reshape((np.asarray(list_ori[i])), (np.asarray(list_ori[i]).shape[0], 1)))
        return  list_selected


    def FeatureExtractorbyObj(self, list_in, feat_index):
        # Extract and Reshape list[obj][feat] according to called feature index
        n = len (list_in)
        list_selected = []
        for i in range(len(list_in)):
            list_selected.append (list_in[i][feat_index])
        return  list_selected

    def ListConversion(self, list_in):
        # Convert list_a[feat][obj] with length of n features into list_b[obj][feat] with length of k objects
        # n_obj = len(list_in[0])
        list_output=[]
        for x in range (len(list_in[0])):
            list_temp = []
            for y in range(len(list_in)):
                list_temp.append(list_in[y][x])
            list_output.append(list_temp)
        return  list_output

    def FlattenAllFeat(self, list_in):
        # Flatten for processing in kalman filter
        list_output = []
        for i in range(len(list_in)):
            for j in range (len(list_in[i])):
                list_output.append(list_in[i][j])
        return  list_output

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

    def compute_IOU(self, F1, F2):
        # Reading the list item in Frame [ymin, ymax, xmin, xmax]
        width1 = F1[3] -F1[2]
        height1 = F1[1] - F1[0]

        width2 = F2[3] - F2[2]
        height2 = F2[1] - F2[0]

        start_x = min (F1[2], F2[2])
        end_x = max ((F1[2]+width1), (F2[2]+width2))
        width = width1 + width2 - (end_x-start_x)

        start_y = min (F1[0], F2[0])
        end_y = max ((F1[0]+height1), (F2[0]+height2))
        height = height1 + height2 - (end_y - start_y)

        if ((width <=0) or (height <= 0)):
            result = 0
        else:
            result = (height*width)/ float((height1*width1)+(height2*width2)-(height*width))

        # Reframe [xmin, ymin, xmax, ymax]
        # x1 = Reframe[0];
        # y1 = Reframe[1];
        # width1 = Reframe[2]-Reframe[0];
        # height1 = Reframe[3]-Reframe[1];
        #
        # x2 = GTframe[0];
        # y2 = GTframe[1];
        # width2 = GTframe[2]-GTframe[0];
        # height2 = GTframe[3]-GTframe[1];

        # endx = max(x1+width1,x2+width2);
        # startx = min(x1,x2);
        # width = width1+width2-(endx-startx);
        #
        # endy = max(y1+height1,y2+height2);
        # starty = min(y1,y2);
        # height = height1+height2-(endy-starty);
        #
        # if width <=0 or height <= 0:
        #     ratio = 0
        # else:
        #     Area = width*height;
        #     Area1 = width1*height1;
        #     Area2 = width2*height2;
        #     ratio = Area*1./(Area1+Area2-Area);
        # # return IOU
        # return ratio,Reframe,GTframe
        return result

    def UpdateTracker(self, detected_per_frame, feat_index, frame_num):
        self.framecount = frame_num
        detections = self.ListConversion(detected_per_frame)  # detections[obj][feat]

        # Create tracks if no tracks vector found
        if (len(self.tracks_per_frame) == 0):  # might wanna add conditions here later, at some point where all objects left scene?
            # print ("no tracks found, assign all detection to tracks")
            # HL: Should to initialize individual track property and history here
            for t, trk in enumerate(detections):
                # obj = KalmanFilterObj(detections[t], dim_x=6, dim_z=4)
                obj = KalmanFilterObj(detections[t], self.framecount, dim_x=7, dim_z=4)
                #  dim_x = for all the information contained by the object
                #  dim_z = for the bounding box coordinate prediction
                self.tracks_per_frame.append(obj)
                self.trackCount += 1
        else:
            print ("Here is length for existing frame track")
            print (len(self.tracks_per_frame))
            # will try combination of features here
            temp = []
            for t, trk in enumerate(self.tracks_per_frame):
                # print (t)
                # print ("HL check trk format now")
                # print (type(self.tracks_per_frame[t]))
                # pos_trk = self.tracks_per_frame[t].predict_track()
                # print (type(self.tracks_per_frame[t]))
                # print (len(self.tracks_per_frame[t]))
                # print (self.tracks_per_frame[t])
                temp.append(trk.retrieve_obj_info())
                # extract all the tracks to retrieve its input features

            # feat_pos = True
            # feat_clr = True
            # feat_lbl = True
            # for t, trk in enumerate(self.tracks_per_frame):
            #     track_feat = trk.retrieve_obj_info()
            #
            #     if (feat_pos):
            #         detection_frame_pos = self.FeatureExtractorbyObj(detections, t)
            #         tracked_frame_pos = track_feat[0]
            #
            #     if (feat_clr):
            #         detection_frame_clr = self.FeatureExtractorbyObj(detections, t)
            #         tracked_frame_clr = track_feat[2]
            #     if (feat_lbl):
            #         detection_frame_clr = self.FeatureExtractorbyObj(detections, t)
            #         tracked_frame_clr = track_feat[2]
            #         # print ("detected pos for this obj")
            #         # print (detection_frame_pos)
            #         # tracked_frame_pos = self.FeatureExtractorbyFeat(track_feat, 0)
            #         # print ("what is this?")
            #         # print (t)
            #         # print (len(track_feat))
            #
            #         # print (track_feat)
            #         # print ("what is this 2?")
            #         # print (self.FeatureExtractorbyFeat(track_feat, 0))
            #
            #         # tracked_frame_pos = self.FeatureExtractorbyObj(self.tracks_per_frame, feat_index[i])
            #     #     tracked_frame_pos = self.FeatureExtractorbyFeat(track_feat[t], 0)
            #     #
            #     #     # print ("Quoi?")
            #     #     # print (feat_index[0])
            #     #     # print (len (self.tracks_per_frame))
            #     #     # tracked_frame_pos = self.FeatureExtractorbyFeat(self.tracks_per_frame[t].retrieve_obj_info(), feat_index[0])
            #     #     pos_trk = self.tracks_per_frame[t].predict_track()
            #     #     # print ("what is the content")
            #     #     # print (self.tracks_per_frame[t].retrieve_obj_info())
            #     #     print ("check detection")
            #     #     print (detection_frame_pos)
            #     #     print (self.FeatureExtractorbyObj(detections, 3))
            #     #     print ("check track")
            #     #     print (tracked_frame_pos)
            #     # # pos_trk = self.tracks_per_frame[t].predict_track()
            #     # # extract all the tracks to retrieve its input features
            #     #
            #
            #     print ("fini un objet")

            for i in range(len(feat_index)):
                # feat = feat_index [i]
                # detected [feat] = self.FeatureExtractorbyFeat(detected_per_frame, feat)
                if (i==0):
                    feat_pos =True
                    detection_frame_pos = self.FeatureExtractorbyFeat(detected_per_frame, feat_index [i])
                    tracked_frame_pos = self.FeatureExtractorbyObj(temp, feat_index [i])

                    # print ("HL check pos format now")
                    # print (type (tracked_frame_pos))
                    # print (len(tracked_frame_pos))
                    # print (tracked_frame_pos)
                    # pos = self.trackers[t].predict()[0]
                elif (i==2):
                    feat_clr = True
                    detection_frame_clr = self.FeatureExtractorbyFeat(detected_per_frame, feat_index [i])
                    tracked_frame_clr = self.FeatureExtractorbyObj(temp, feat_index [i])
                elif (i==3):
                    feat_lbl = True
                    detection_frame_lbl = self.FeatureExtractorbyFeat(detected_per_frame, feat_index [i])
                    tracked_frame_lbl = self.FeatureExtractorbyObj(temp, feat_index [i])
                else:
                    feat_ctr = True
                    detection_frame_ctr = self.FeatureExtractorbyFeat(detected_per_frame, feat_index[i])


            # Calculate cost using sum of square distance between tracks and detection
            # Note that this particular cost is computed for box coordinate feature
            if (feat_pos):
                cost_pos = np.zeros(shape=(len(tracked_frame_pos), len(detection_frame_pos)))  # Cost matrix
                diff_pos = np.zeros(shape=(len(tracked_frame_pos), len(detection_frame_pos)))

                # iou_matrix = np.zeros((len(detect_list), len(track_list)), dtype=np.float32)

                # for d, det in enumerate(detect_list):
                #     for t, trk in enumerate(track_list):
                #         iou_matrix[d, t] = self.compute_IOU(det, trk)

                for i in range(len(tracked_frame_pos)):
                    # norm_track = [tracked_frame_pos[i][0]/self.im_height, tracked_frame_pos[i][1]/self.im_height, tracked_frame_pos[i][2]/self.im_width, tracked_frame_pos[i][3]/self.im_width]
                    for j in range(len(detection_frame_pos)):
                        # norm_detected = [detection_frame_pos[j][0] / self.im_height, detection_frame_pos[j][1] / self.im_height,
                        #                  detection_frame_pos[j][2] / self.im_width, detection_frame_pos[j][3] / self.im_width]

                        # diff_pos = tracked_frame_pos[i] - detection_frame_pos[j]
                        # # diff = norm_track - norm_detected
                        # distance_pos = np.sqrt(diff_pos[0][0] * diff_pos[0][0] + diff_pos[1][0] * diff_pos[1][0])
                        # cost_pos[i][j] = (0.5) * distance_pos
                        cost_pos[i, j] = 1-self.compute_IOU(tracked_frame_pos[i], np.reshape(detection_frame_pos[j], (4)).tolist()) # low cost indicates high IOU

                # print ("Check this is IOU")
                # print(cost_pos)
                # row_ind_pos, col_ind_pos = linear_sum_assignment(cost_pos)
                # matched_indices_pos = np.concatenate((row_ind_pos.reshape((row_ind_pos.shape[0], 1)), col_ind_pos.reshape((col_ind_pos.shape[0], 1))), axis=1)
                #
                # print (row_ind_pos)
                # print (col_ind_pos)
                # print (matched_indices_pos)
########################################################################################
            if (feat_lbl):
                cost_lbl = np.zeros(shape=(len(tracked_frame_lbl), len(detection_frame_lbl)))  # Cost matrix
                diff_lbl = np.zeros(shape=(len(tracked_frame_lbl), len(detection_frame_lbl)))
                for i in range(len(tracked_frame_lbl)):
                    for j in range(len(detection_frame_lbl)):
                        # diff_lbl = tracked_frame_lbl[i] - detection_frame_lbl[j]
                        # distance_lbl = np.sqrt(diff_lbl[0][0] * diff_lbl[0][0] + diff_lbl[1][0] * diff_lbl[1][0])
                        # cost_lbl[i][j] = (0.5) * distance_lbl
                        if (tracked_frame_lbl[i][0]==detection_frame_lbl[j][0]):
                            # cost_lbl[i, j] = 1- ((tracked_frame_lbl[i][1])*(detection_frame_lbl[j][1]))
                            cost_lbl[i, j] = 1 - 0.5* (((tracked_frame_lbl[i][1])+(detection_frame_lbl[j][1])))
                        else:
                            cost_lbl[i, j] = 1
                            # cost_lbl[i, j] = (tracked_frame_lbl[i][1]) * (detection_frame_lbl[j][1])
                # print ("Check this is Label")
                # print(cost_lbl)
                # cost_lbl[i][j] =

                # row_ind_lbl, col_ind_lbl = linear_sum_assignment(cost_lbl)
                # matched_indices_lbl = np.concatenate((row_ind_lbl.reshape((row_ind_lbl.shape[0], 1)), col_ind_lbl.reshape((col_ind_lbl.shape[0], 1))), axis=1)
                # print (row_ind_lbl)
                # print (col_ind_lbl)
                # print (matched_indices_lbl)
########################################################################################

            if (feat_clr):
                cost_clr = np.zeros(shape=(len(tracked_frame_clr), len(detection_frame_clr)))  # Cost matrix
                for i in range(len(tracked_frame_clr)):
                    for j in range(len(detection_frame_clr)):
                        cost_clr[i][j] = cv2.compareHist(tracked_frame_clr[i], detection_frame_clr[j], cv2.HISTCMP_BHATTACHARYYA)
                # print(cost_clr)
                # print ("Check this is color histogram distance")
                # print(cost_clr)

                # row_ind_clr, col_ind_clr = linear_sum_assignment(cost_clr)
                # matched_indices_clr = np.concatenate((row_ind_clr.reshape((row_ind_clr.shape[0], 1)), col_ind_clr.reshape((col_ind_clr.shape[0], 1))),axis=1)
                # print (row_ind_clr)
                # print (col_ind_clr)
                # print (matched_indices_clr)
########################################################################################
            print ("matching from all three features")
            cost_combined = cost_pos + cost_lbl + cost_clr
            print (cost_combined)
            row_ind, col_ind = linear_sum_assignment(cost_combined)
            matched_indices = np.concatenate((row_ind.reshape((row_ind.shape[0], 1)), col_ind.reshape((col_ind.shape[0], 1))), axis=1)
            print (row_ind)
            print (col_ind)
            print (matched_indices)

            #currently using position feature only
            unmatched_detections = []
            for d, det in enumerate(detection_frame_pos):
                if (d not in matched_indices[:, 1]):
                    unmatched_detections.append(d)
            unmatched_trackers = []
            for t, trk in enumerate(self.tracks_per_frame):
                if (t not in matched_indices[:, 0]):
                    unmatched_trackers.append(t)

            # remove matches with too high cost
            matches = []
            for m in matched_indices:
                if (cost_combined[m[0], m[1]] < self.dist_thresh):
                    matches.append(m.reshape(1, 2))
                else:
                    unmatched_detections.append(m[1])
                    unmatched_trackers.append(m[0])

            if (len(matches) == 0):
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)

            print ("unmatched detection")
            print (unmatched_detections)
            print ("unmatched trackers")
            print (unmatched_trackers)
            print ("matched")
            print (matches)

            # Track management and updates
            # Initialize new track for new detection
            for t, trk in enumerate(unmatched_detections):
                obj = KalmanFilterObj(detections[t], self.framecount, dim_x=7, dim_z=4)
                #  dim_x = for all the information contained by the object
                #  dim_z = for the bounding box coordinate prediction
                self.tracks_per_frame.append(obj)
                self.trackCount += 1

            # self.active_tracks = self.tracks_per_frame

               # Update matched track and unmatched track
            for t, trk in enumerate((self.tracks_per_frame)):
                # for t, trk in enumerate(self.tracks_per_frame[feat_index]):
                if (t not in unmatched_trackers):
                    # print ("all pair matches, update!")
                    d_detected = matches[np.where(matches[:, 1] == t)[0], 1]
                    t_tracked = matches[np.where(matches[:, 1] == t)[0], 0]
                    # Update individual object property in the track list

                    if (d_detected.size and t_tracked.size): #just not make sure the array is not empty
                        # trk.update()
                        trk.update(detections, t_tracked[0], d_detected[0]) #kalmanfilter2track
                else:
                    trk.predict_track()

            self.active_tracks = []

            # Removal of dead track
            for t, trk in enumerate((self.tracks_per_frame)):
                if (t not in unmatched_trackers):
                    self.active_tracks.append(trk)
                else:
                    if (trk.time_since_update < self.max_trace_length):
                        self.active_tracks.append(trk)
            # print ("did I delete the dead track")
            # print (len(self.tracks_per_frame))
            del self.tracks_per_frame[:]

            self.tracks_per_frame = self.active_tracks

        return (self.active_tracks)

