This is a multi-step process. In order to identify objects in an image, you need to:
1. Slice a large image up into multiple smaller sub-images. By default these are 450x450px.
2. Create a JSON file that stores the crater positions within each sub-image.
3. Create a tensorflow file that contains all of the training image data and metadata (and one for evaluation as well).
4. Begin training, starting with reading in a tensorflow config file (which points to the created training and evaluation files).
5. Wait for a day or two for the training to happen. If you want to see progress, run eval.py, which returns a score between 0 and 1 on a set of images.
6. Freeze a version of the neural network to use as a tool for identifying objects in the image.
7. Use that frozen neural network to find objects in an image.


Slice an image and return a JSON file with data for each sub-image.
If craters_file and coordinates are specified, the JSON will include the locations of craters in each sub-image.
coordinates is the left,top,right,bottom coordinates of the image in degrees. It will then slice the image, apply the correct coordinates
to each sub-image, and find all of the provided craters that fit in that sub-image.
 >python image_slicer.py --image_file=images/WAC_GLOBAL_E300N0450_100M.png --coordinates=0,0,90,90 --craters_file=stuart_moon_craters.csv

If you want to create a JSON file for rafaelg's data, which is pulled from a database, use:
 >create_db_training_data.py

Modify the ALL_DATA_FILES variable in create_training_data.py to reference all of your json files,
and the TRAINING_OUTPUT_PATH and EVALUATION_OUTPUT_PATH if you want.
 >vim create_training_data.py

Then run the file to create the training and evaluation files
 >python create_training_data.py

Modify config/cosmoquest-trainer.config, and near the bottom of the file, change "cosmoquest/config/train.record" and "cosmoquest/config/eval.record"
to your TRAINING_OUTPUT_PATH and EVALUATION_OUTPUT_PATH
 >vim config/cosmoquest-trainer.config

Run training. This can take a long time. You want the loss to be pretty consistently under 1. The more consistent, the better. This can take 100000 iterations or more.
 >python train.py --logtostderr --train_dir=cosmoquest/training --pipeline_config_path=cosmoquest/config/cosmoquest-trainer.config

Run evaluation
 >python eval.py --logtostderr --eval_dir=cosmoquest/eval --pipeline_config_path=cosmoquest/config/cosmoquest-trainer.config --checkpoint_dir="cosmoquest/training"

In order to run the Neural Network on an image, you need to freeze a version of the network
 >python export_inference_graph.py --input_type image_tensor --pipeline_config_path=cosmoquest\config\cosmoquest-trainer.config --trained_checkpoint_prefix=cosmoquest\training\model.ckpt-7767 --output_directory=cosmoquest\training\frozen_graph

Modify IMAGE_DIRECTORY and OUTPUT_DIRECTORY in find_craters_in_image.py to point to the directory holding your images and where you want the output to be saved.
 >vim find_craters_in_image.py

Find the craters in a set of images
 >python find_craters_in_image.py



Note that object_detection is a clone of the research/object_detection directory in https://github.com/tensorflow/models.git

If you want to try a different neural network arrangement, try using config/cosmoquest-trainer.config, or modify any network in object_detection/samples/configs/.