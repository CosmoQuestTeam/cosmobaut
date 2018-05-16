import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


IMAGE_DIRECTORY = "cosmoquest/test_images/test_images/"
OUTPUT_DIRECTORY = 'cosmoquest/test_images/test_images/verify/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'cosmoquest/training/frozen_graph/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'cosmoquest/config/object_labels.pbtxt'

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')



NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        # text_format.Parse(serialized_graph, od_graph_def)
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    image_data = image.getdata()
    new_image_data = []
    for pixel in image_data:
        if isinstance(pixel, int):
            new_image_data.append([pixel, pixel, pixel])
        else:
            new_image_data.append([pixel[0], pixel[0], pixel[0]])

    return np.array(new_image_data).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (8, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session(config=config) as sess:
            # with tf.Session() as sess:
            # Get handles to input and output tensors
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
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


image_paths = [os.path.join(IMAGE_DIRECTORY, f) for f in os.listdir(IMAGE_DIRECTORY) if
               os.path.isfile(os.path.join(IMAGE_DIRECTORY, f))]
for image_path in image_paths:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    scores = np.squeeze(output_dict['detection_scores'])
    boxes = np.squeeze(output_dict['detection_scores'])
    box_list = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            box_list.append(output_dict['detection_boxes'][i])

    print(output_dict['num_detections'])
    output_dict['detection_boxes'] = np.array(box_list)

    if len(output_dict['detection_boxes']) > 0:
        # Visualization of the results of a detection.
        vis_util.draw_bounding_boxes_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            color='red',
            thickness=2)

    if not os.path.exists(os.path.dirname(os.path.realpath(image_path))):
        os.makedirs(image_path)
    vis_util.save_image_array_as_png(image_np,
                                     OUTPUT_DIRECTORY + os.path.splitext(os.path.basename(image_path))[0] + '.png')
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
