# Reads in crater data from the database and writes it to a file in a format appropriate for training a TensorFlow AI

import MySQLdb
import tensorflow
import urllib.request
import os
import json
from PIL import Image
from utils import dataset_util

OUTPUT_PATH = "cosmoquest/config/train.record"
IMAGE_FORMAT = b'png'
DB_NAME = "cosmoquest"
DB_USER = "root"
DB_PASSWORD = "guest"
DB_HOST = "localhost"

database = MySQLdb.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASSWORD, db=DB_NAME).cursor()

all_training_data = []
database.execute("""SELECT id, 450, 450, file_location, application_id FROM images WHERE application_id = 1 AND id in (SELECT distinct image_id FROM marks WHERE user_id = 106) ORDER BY rand()""")
#database.execute("""SELECT id, 450, 450, file_location FROM images WHERE application_id = 1 AND done = true AND id != 60639 AND details = 'foo' ORDER BY rand()""")
images = database.fetchall()
# Collect the training data from each image
for image in images:
    image_id = image[0]
    image_width = image[1]
    image_height = image[2]
    #remote_file_location = image[3]
    application_id = image[4]
    local_file_location = "images/" + str(application_id) + "/" + str(image_id) + ".png"
    local_directory = os.path.dirname(os.path.realpath(local_file_location))

    #if not os.path.exists(local_directory):
    #    os.makedirs(local_directory)

    #try:
        #if not os.path.exists(local_file_location):
        #    urllib.request.urlretrieve(remote_file_location, local_file_location)

        #im = Image.open(local_file_location)
        #local_file_location = str.replace(local_file_location, "png", "jpeg")
        #im.save(local_file_location, "JPEG")

    #fp = open(local_file_location, 'rb')
    #file_contents = fp.read()

    x_mins = []
    x_maxes = []
    y_mins = []
    y_maxes = []
    classes_text = []
    classes = []
    craters = []

    database.execute("""SELECT x, y, diameter FROM marks WHERE user_id = 106 AND type = 'crater' AND image_id = %s""", (image_id,))
    marks = database.fetchall()
    for mark in marks:
        x = mark[0]
        y = mark[1]
        diameter = mark[2]
        new_crater = {"left": max(0, round(x - diameter / 2, 1)),
                      "top": max(0, round(y - diameter / 2, 1)),
                      "right": min(450, round(x + diameter / 2, 1)),
                      "bottom": min(450, round(y + diameter / 2, 1))}
        craters.append(new_crater)
        # The bounding box coordinates need to be a percent relative to the size of the image
        '''x_mins.append(max(0, (x - diameter / 2) / image_width))
        x_maxes.append(min(1, (x + diameter / 2) / image_width))
        y_mins.append(max(0, (y - diameter / 2) / image_height))
        y_maxes.append(min(1, (y + diameter / 2) / image_height))
        classes_text.append(b'crater')
        classes.append(1)
    
    training_data = tensorflow.train.Example(features=tensorflow.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(str.encode(local_file_location)),
        'image/source_id': dataset_util.bytes_feature(str.encode(local_file_location)),
        'image/encoded': dataset_util.bytes_feature(file_contents),
        'image/format': dataset_util.bytes_feature(IMAGE_FORMAT),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxes),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxes),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    all_training_data.append(training_data)'''
    all_training_data.append({"file_location": local_file_location, "width": image_width,
                              "height": image_height, "craters": craters})
    #except:
    #    print("Error reading file: " + local_file_location)

# Write those images to a file in the format that TensorFlow wants
#writer = tensorflow.python_io.TFRecordWriter(OUTPUT_PATH)
#for training_data in all_training_data:
    #writer.write(training_data.SerializeToString())
text = json.dumps(all_training_data, indent=4)
file = open(os.path.join("images/", "sub_image_data.json"), "w")
file.write(text)
file.close()