import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
from PIL import Image


class Detector:
    def __init__(self):
        pass



    def _get_detection_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # grabbing the inference graph
            with tf.gfile.GFile('./model_parts/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # grabbing the label map
        category_index = label_map_util.create_category_index_from_labelmap('./model_parts/labelmap.pbtxt', use_display_name=True)
        return detection_graph, category_index

    #Turning the image into an array.
    def _image_to_array(self,image_path):
        image = Image.open(image_path)
        (img_width, img_height) = image.size

        return np.array(image.getdata()).reshape(img_height, img_width, 3).astype(np.uint8)
    #run inference graph over a image labeling any identified objects
    def _detect_image(self, image, graph):
      with graph.as_default():
        with tf.Session() as sess:

          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:

            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)

            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: image})

          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.int64)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
    #run the detector over a image and saved the output image

    def detect(self,image_path,output_path):
      image_size = (14,14)
      detection_graph, category_index = self._get_detection_graph()

      image = Image.open(image_path)
      image_np = self._image_to_array(image_path)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      output_dict = self._detect_image(image_np_expanded, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=6)
      plt.figure(figsize=image_size)
      plt.imshow(image_np)
      plt.savefig(output_path)
