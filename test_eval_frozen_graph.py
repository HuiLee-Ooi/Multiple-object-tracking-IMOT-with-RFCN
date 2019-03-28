import argparse
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
    #                     help="Frozen model file to import")
    # args = parser.parse_args()
    #
    # # We use our "load_graph" function
    # graph = load_graph(args.frozen_model_filename)

    detection_graph = load_graph("/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_rfcn_resnet101_mio/train/output_fin.pb")
    # detection_graph = load_graph("/home/huooi/HL_Proj/PycharmProjects/models/object_detection/HL_testing/train/faster_rcnn_resnet101_coco_2017_11_08/frozen_inference_graph.pb")
    # We can verify that we can access the list of operations in the graph
    for op in detection_graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    x = detection_graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    y = detection_graph.get_tensor_by_name('prefix/Accuracy/predictions:0')

    with tf.Session(graph=detection_graph) as sess:
        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        image = Image.open("/home/huooi/HL_Dataset/lena.jpg")
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # # Each box represents a part of the image where a particular object was detected.
        # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # # Each score represent how level of confidence for each of the objects.
        # # Score is shown on the result image, together with the class label.
        # scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #
        # # Actual detection.
        # (boxes, scores, classes, num_detections) = sess.run(
        #     [boxes, scores, classes, num_detections],
        #     feed_dict={image_tensor: image_np_expanded})
        #
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image)
        plt.show()
        # print("hello")

        # y_out = sess.run(y, feed_dict={
        #     x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]]  # < 45
        # })

        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        # print(y_out)  # [[ False ]] Yay, it works!
