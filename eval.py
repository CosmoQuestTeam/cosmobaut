# Trains a Tensorflow instance based on the provided data.
# This code is copied from object_detection/train.py. It's pretty much unreadable garbage, but it usually works.

import functools
import os
import tensorflow as tf

from object_detection import evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_boolean('run_once', False, 'Option to only run a single pass of '
                     'evaluation. Overrides the `max_evals` parameter in the '
                     'provided config.')
FLAGS = flags.FLAGS


def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  if FLAGS.pipeline_config_path:
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
    tf.gfile.Copy(FLAGS.pipeline_config_path,
                  os.path.join(FLAGS.eval_dir, 'pipeline.config'),
                  overwrite=True)
  else:
    configs = config_util.get_configs_from_multiple_files(
        model_config_path=FLAGS.model_config_path,
        eval_config_path=FLAGS.eval_config_path,
        eval_input_config_path=FLAGS.input_config_path)
    for name, config in [('model.config', FLAGS.model_config_path),
                         ('eval.config', FLAGS.eval_config_path),
                         ('input.config', FLAGS.input_config_path)]:
      tf.gfile.Copy(config,
                    os.path.join(FLAGS.eval_dir, name),
                    overwrite=True)

  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']
  if FLAGS.eval_training_data:
    input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=False)

  def get_next(config):
    return dataset_util.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)

  label_map = label_map_util.load_labelmap(input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                     FLAGS.checkpoint_dir, FLAGS.eval_dir)


if __name__ == '__main__':
  tf.app.run()
