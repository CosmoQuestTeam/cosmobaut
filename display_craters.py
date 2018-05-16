import json
import tensorflow
import os
import random
from utils import dataset_util
import matplotlib.pyplot as plt

OUTPUT_DIRECTORY = "marked_images/"

json_data = open("images/WAC_GLOBAL_E300N3150_100M/sub_image_data.json").read()
image_data = json.loads(json_data)
image_data.reverse()
for image in image_data:
    rectangles = []
    for crater in image["craters"]:
        color = "red"
        rectangles.append(plt.Rectangle((crater["left"],
                                        crater["top"]),
                                        crater["right"] - crater["left"],
                                        crater["bottom"] - crater["top"],
                                        color=color, fill=False, lw=1))
    fig = plt.figure()
    fig.set_size_inches((9.375, 9.375))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img = plt.imread(image["file_location"])
    ax.imshow(img, cmap='gray', aspect='equal')
    for rectangle in rectangles:
        ax.add_patch(rectangle)
    output_file = str.replace(image["file_location"], "images/", OUTPUT_DIRECTORY)
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(output_file, dpi=48)
    plt.close()

'''import MySQLdb
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT_PATH = "cosmoquest/config/eval.record"
DB_NAME = "cosmoquest"
DB_USER = "root"
DB_PASSWORD = "guest"
DB_HOST = "localhost"

database = MySQLdb.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASSWORD, db=DB_NAME).cursor()

all_training_data = []

database.execute("""SELECT id, 450, 450, file_location FROM images WHERE application_id = 1 AND id in (SELECT distinct image_id FROM marks WHERE user_id = 106) ORDER BY rand()""")
images = database.fetchall()

for image in images:
    image_id = image[0]
    image_width = image[1]
    image_height = image[2]
    remote_file_location = image[3]
    local_file_location = str.replace(remote_file_location, "https://s3.amazonaws.com/cosmoquest/data/mappers/moon/", "rafael/")
    print(local_file_location)
    local_directory = os.path.dirname(os.path.realpath(local_file_location))

    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

        #if not os.path.exists(local_file_location):
            #urllib.request.urlretrieve(remote_file_location, local_file_location)

        database.execute("""SELECT x, y, diameter FROM marks WHERE user_id = 106 AND image_id = %s""", (image_id,))
        marks = database.fetchall()
        print(len(marks))

        circles = []
        for mark in marks:
            color = "red"
            circles.append(plt.Circle((mark[0], mark[1]), radius=mark[2] / 2, color=color, fill=False, lw=4.5))

        fig = plt.figure()
        fig.set_size_inches((9.375, 9.375))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = plt.imread(remote_file_location)
        ax.imshow(img, cmap='gray', aspect='equal')
        for circle in circles:
            ax.add_patch(circle)
        plt.savefig(local_file_location, dpi=48)
        plt.close()'''