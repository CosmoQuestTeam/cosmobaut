# Reads in crater data from the database and writes it to a file in a format appropriate for training a TensorFlow AI

import json
import tensorflow
import os
import random
from utils import dataset_util

TRAINING_OUTPUT_PATH = "../config/train.record"
EVALUATION_OUTPUT_PATH = "../config/eval.record"
ALL_DATA_FILES = ["images/1/sub_image_data.json",
                  "images/WAC_GLOBAL_E300N0450_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300N1350_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300N2250_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300N3150_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300S0450_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300S1350_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300S2250_100M/sub_image_data.json",
                  "images/WAC_GLOBAL_E300S3150_100M/sub_image_data.json"]
IMAGE_FORMAT = b'png'

all_training_data = []
all_evaluation_data = []
evaluation_images = []
image_data = []

for data_file in ALL_DATA_FILES:
    json_data = open(data_file).read()
    image_data += json.loads(json_data)

random.shuffle(image_data)
count = 0
# Collect the training data from each image
for image in image_data:
    if count % 1000 == 0:
        print(count)
    count += 1
    try:
        fp = open(image["file_location"], 'rb')
        file_contents = fp.read()

        x_mins = []
        x_maxes = []
        y_mins = []
        y_maxes = []
        classes_text = []
        classes = []

        for crater in image["craters"]:
            # The bounding box coordinates need to be a percent relative to the size of the image
            x_mins.append(max(0, crater["left"] / image["width"]))
            x_maxes.append(min(1, crater["right"] / image["width"]))
            y_mins.append(max(0, crater["top"] / image["height"]))
            y_maxes.append(min(1, crater["bottom"] / image["height"]))
            classes_text.append(b'crater')
            classes.append(1)

        training_data = tensorflow.train.Example(features=tensorflow.train.Features(feature={
            'image/width': dataset_util.int64_feature(image["width"]),
            'image/height': dataset_util.int64_feature(image["height"]),
            'image/filename': dataset_util.bytes_feature(str.encode(image["file_location"])),
            'image/source_id': dataset_util.bytes_feature(str.encode(image["file_location"])),
            'image/encoded': dataset_util.bytes_feature(file_contents),
            'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
            'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxes),
            'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxes),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        if random.random() < 0.03:
            evaluation_images.append(image["file_location"])
            all_evaluation_data.append(training_data)
        else:
            all_training_data.append(training_data)
    except:
        print("Error with image: " + image["file_location"])

# Write those images to a file in the format that TensorFlow wants
if not os.path.exists(os.path.dirname(os.path.realpath(TRAINING_OUTPUT_PATH))):
    os.makedirs(os.path.dirname(os.path.realpath(TRAINING_OUTPUT_PATH)))
if not os.path.exists(os.path.dirname(os.path.realpath(EVALUATION_OUTPUT_PATH))):
    os.makedirs(os.path.dirname(os.path.realpath(EVALUATION_OUTPUT_PATH)))

writer = tensorflow.python_io.TFRecordWriter(TRAINING_OUTPUT_PATH)
for training_data in all_training_data:
    writer.write(training_data.SerializeToString())
writer = tensorflow.python_io.TFRecordWriter(EVALUATION_OUTPUT_PATH)
for evaluation_data in all_evaluation_data:
    writer.write(evaluation_data.SerializeToString())

text = json.dumps(evaluation_images, indent=4)
file = open("unused_images.json", "w")
file.write(text)
file.close()
