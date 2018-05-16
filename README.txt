Getting the error:
	ValueError: cannot reshape array of size 202500 into shape (450,450,3)
means that the image that you are inputting into the neural network is B&W and needs to be color.


// Create record files to hold image and crater data
cosmo_prepare.py


// Run training
python train.py --logtostderr --train_dir=cosmoquest/training --pipeline_config_path=cosmoquest/config/cosmoquest-trainer.config

// Run evaluation
python evaluator.py --logtostderr --eval_dir=cosmoquest/eval --pipeline_config_path=cosmoquest/config/cosmoquest-trainer.config --checkpoint_dir="cosmoquest/training"


// In order to run the Neural Network on an image, you need to freeze a version of the network
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=cosmoquest\config\cosmoquest-trainer.config --trained_checkpoint_prefix=cosmoquest\training\model.ckpt-7767 --output_directory=cosmoquest\training\frozen_graph