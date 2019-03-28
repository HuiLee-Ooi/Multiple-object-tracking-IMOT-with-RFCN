# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/usagers/huper/dev/data/MIO-TCD-Localization \
        --output_path=/usagers/huper/dev/data/MIO-TCD-Localization/mio.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import xml.etree.ElementTree



flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/detrac_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val']

def main(_):

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  data_dir = FLAGS.data_dir

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  examples_path = os.path.join(data_dir,  FLAGS.set + '_examples.txt')
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir + 'annotations.txt')
  if FLAGS.set == "val":
      annotations_dir = os.path.join(data_dir, "val/annotations")
  if FLAGS.set == "train":
      annotations_dir = os.path.join(data_dir, "train/annotations")

  # examples_list = dataset_util.read_examples_list(examples_path)
  print(os.listdir(annotations_dir))
  # read every xml file one by one
  for file_name in os.listdir(annotations_dir):
    print(file_name)
    # set the image folder
    image_folder = "/store/datasets/UA-Detrac/" + FLAGS.set + "/images/" + file_name.strip(".xml") + "/"

    xml_file = xml.etree.ElementTree.parse(annotations_dir + "/" + file_name).getroot()
    # see each frame one by one
    counter = 0
    for frame in xml_file.findall('frame'):

      num = frame.get('num')
      # for decimated version:
      counter += 1
      if counter != 1:
          if counter == 10:
              counter = 0
          continue

      image_file_name = image_folder + "img" + str(num).zfill(5) + ".jpg"
      objects = (frame.find('target_list')).findall('target')

      detections = []

      # see each object of the frame one by one, store them in detections
      for object in objects:
          box = object.find('box')
          left = box.get('left')
          top = box.get('top')
          width = box.get('width')
          height = box.get('height')

          attribute = object.find('attribute')
          type = attribute.get('vehicle_type')
          detections.append((type, (float(left), float(top), float(left) + float(width), float(top) + float(height))))

      img_path = image_file_name
      with tf.gfile.GFile(img_path, 'rb') as fid:
          encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = PIL.Image.open(encoded_jpg_io)
      if image.format != 'JPEG':
          raise ValueError('Image format not JPEG')
      key = hashlib.sha256(encoded_jpg).hexdigest()

      width = 960
      height = 540

      xmin = []
      ymin = []
      xmax = []
      ymax = []
      classes = []
      classes_text = []
      truncated = []
      poses = []
      difficult_obj = []
      for obj in detections:
          difficult = False

          difficult_obj.append(int(difficult))

          xmin.append(float(obj[1][0]) / width)
          ymin.append(float(obj[1][1]) / height)
          xmax.append(float(obj[1][2]) / width)
          ymax.append(float(obj[1][3]) / height)
          classes_text.append(obj[0].encode('utf8'))
          classes.append(label_map_dict[obj[0]])

      example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(
              image_file_name.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(
              image_file_name.encode('utf8')),
          'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
          'image/object/difficult': dataset_util.int64_list_feature(0),
          'image/object/truncated': dataset_util.int64_list_feature(0),
          'image/object/view': dataset_util.bytes_feature('Left'.encode('utf8')),
      }))
      writer.write(example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  tf.app.run()
