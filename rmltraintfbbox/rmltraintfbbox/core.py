"""
Author(s):      Nihal Dhamani (nihaldhamani@gmail.com), 
                Carson Schubert (carson.schubert14@gmail.com)
Date Created:   12/06/2019

Core command group and commands for TF Bounding Box plugin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment
import os
import click 
import io
import sys
import yaml
import importlib
import re
import glob
import json
import traceback
import rmltraintfbbox.validation.utils as utils
from contextlib import ExitStack
from pathlib import Path
from datetime import datetime
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import cli_spinner, user_selects, user_input
from ravenml.utils.plugins import raise_parameter_error
from rmltraintfbbox.utils.helpers import prepare_for_training, download_model_arch
from rmltraintfbbox.utils.exporter import export_inference_graph
from rmltraintfbbox.validation.stats import BoundingBoxEvaluator
from google.protobuf import text_format
from matplotlib import pyplot as plt
import time

# regex to ignore 0 indexed checkpoints
checkpoint_regex = re.compile(r'model.ckpt-[1-9][0-9]*.[a-zA-Z0-9_-]+')

### OPTIONS ###

### COMMANDS ###
@click.group(help='TensorFlow Object Detection with bounding boxes.')
@click.pass_context
def tf_bbox(ctx):
    pass
    
@tf_bbox.command(help='Train a model.')
@pass_train
@click.pass_context
def train(ctx: click.Context, train: TrainInput):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which execution will fail as this means 
    # the user did not pass a config. see ravenml core file train/commands.py for more detail
    
    # NOTE: after training, you must create an instance of TrainOutput and return it
    # import necessary libraries
    cli_spinner("Importing TensorFlow...", _import_od)

    ## SET UP CONFIG ##
    config = train.plugin_config
    metadata = train.plugin_metadata
    comet = config.get('comet')

    if not config['verbose']:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # set base directory for model artifacts 
    base_dir = train.artifact_path
 
    # load model choices from YAML
    models = {}
    models_path = os.path.dirname(__file__) / Path('utils') / Path('model_info.yml')
    with open(models_path, 'r') as stream:
        try:
            models = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # prompt for model selection if not in config
    model_name = config.get('model')
    model_name = model_name if model_name else user_selects('Choose model', models.keys())
    # grab fields and add to metadata
    try:
        model = models[model_name]
    except KeyError as e:
        hint = 'model name, model is not supported by this plugin.'
        raise_parameter_error(model_name, hint)
    
    # extract information and add to metadata
    model_type = model['type']
    model_url = model['url']
    metadata['architecture'] = model_name
    
    # download model arch
    arch_path = download_model_arch(model_url, train.plugin_cache)

    # prepare directory for training/prompt for hyperparams
    if not prepare_for_training(train.plugin_cache, train.artifact_path, train.dataset.path, 
        arch_path, model_type, metadata, train.plugin_config):
        ctx.exit('Training cancelled.')

    model_dir = os.path.join(base_dir, 'models/model')
    train_dir = os.path.join(model_dir, 'train')
    eval_dir = os.path.join(model_dir, 'eval')
    pipeline_config_path = os.path.join(model_dir, 'pipeline.config')

    experiment = None
    if comet:
        experiment = Experiment(workspace='seeker-rd', project_name='bounding-box')
        experiment.set_name(comet)
        experiment.log_parameters(metadata['hyperparameters'])
        experiment.set_git_metadata()
        experiment.set_os_packages()
        experiment.set_pip_packages()
        experiment.log_asset(pipeline_config_path)

    # get number of training steps
    num_train_steps = int(metadata['hyperparameters']['train_steps'])

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']
    eval_input_config = configs['eval_input_config']
    eval_config = configs['eval_config']

    detection_model = model_builder.build(model_config=model_config, is_training=True)
    
    #create tf.data.Dataset()
    train_input = inputs.train_input(train_config, train_input_config, model_config, model=detection_model)
    eval_input = inputs.eval_input(eval_config, eval_input_config, model_config, model=detection_model)

    train_input_iterator = iter(train_input)

    global_step = tf.Variable(
        0, trainable=False, dtype=tf.int64, name='global_step',
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    optimizer, (learning_rate,) = od.builders.optimizer_builder.build(
        train_config.optimizer, global_step=global_step)

    if callable(learning_rate):
      learning_rate_fn = learning_rate
    else:
      learning_rate_fn = lambda: learning_rate

    #restore from checkpoint
    load_fine_tune_checkpoint(detection_model, train_config.fine_tune_checkpoint, 
                                        train_config.fine_tune_checkpoint_type,
                                        train_config.fine_tune_checkpoint_version,
                                        train_input, 
                                        train_config.unpad_groundtruth_tensors)

    @tf.function
    def train_step(detection_model, train_input_iterator, optimizer, learning_rate_fn, global_step):

        features, labels = train_input_iterator.next()
        loss = model_lib2.eager_train_step(detection_model, features,
                            labels, train_config.unpad_groundtruth_tensors,
                            optimizer, learning_rate_fn(),
                            add_regularization_loss=True,
                            clip_gradients_value=None,
                            global_step=global_step)
        global_step.assign_add(1)
        return loss

    #@tf.function
    def evaluate(detection_model, configs, eval_input, global_step):

        text_trap = io.StringIO()
        sys.stdout = text_trap
        sys.stderr = text_trap
        
        metrics = model_lib2.eager_eval_loop(detection_model, configs, eval_input, global_step=global_step)
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        click.echo(f'Evaluation loss: {metrics["Loss/total_loss"]}, Evaluation mAP: {metrics["DetectionBoxes_Precision/mAP"]}')

        return metrics

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=detection_model)
    manager = tf.train.CheckpointManager(checkpoint, directory=model_dir, max_to_keep=5)
      
    with ExitStack() as stack:
        if comet:
            stack.enter_context(experiment.train())
        click.echo('Training model...')

        start = time.time()
        #main training loop
        for step in range(1, num_train_steps+1):

            loss = train_step(detection_model, train_input_iterator, optimizer, learning_rate_fn, global_step)
            
            if step % 100 == 0:
                print(f'Training loss at step {step}: {loss}')
                manager.save()

            if step % 500 == 0:
                evaluate(detection_model, configs, eval_input, global_step)
                

        training_time = time.time() - start
        
        click.echo(f'Training complete. Took {training_time} seconds.')

    # final metadata and return of TrainOutput object
    metadata['date_completed_at'] = datetime.utcnow().isoformat() + "Z"

    # get extra config files
    extra_files, final_ckpt = _get_paths_for_extra_files(base_dir)

    if comet:
        experiment.log_asset(model_path)

    # TODO: make evaluation optional
    if config['evaluate']:
        click.echo("Evaluating model...")
        with ExitStack() as stack:
            if comet:
                stack.enter_context(experiment.validate())
            # path to label_map.pbtxt
            label_path = str(train.dataset.path / 'label_map.pbtxt')
            test_path = str(train.dataset.path / 'test')
            output_path = str(base_dir / 'validation')
            os.mkdir(output_path)

            image_dataset = utils.get_image_dataset(test_path)
            truth_data = list(utils.gen_truth_data(test_path))

            category_index = od.utils.label_map_util.create_category_index_from_labelmap(label_path)
            evaluator = BoundingBoxEvaluator(category_index)

            for (i, (bbox, centroid, z)), image in zip(enumerate(truth_data), image_dataset):
                true_shape = tf.expand_dims(tf.convert_to_tensor(image.shape), axis=0)
                truth = {'groundtruth_boxes': bbox, 'groundtruth_classes': 1}
                start = time.time()
                output = detection_model.call(tf.expand_dims(image, axis=0))
                inference_time = time.time() - start
                evaluator.add_single_result(output, true_shape, inference_time, bbox, centroid)
                drawn_img = od.utils.visualization_utils.draw_bounding_boxes_on_image_tensors(tf.cast(tf.expand_dims(image, axis=0), dtype=tf.uint8), output['detection_boxes'], tf.cast(output['detection_classes'] + 1, dtype=tf.int32), output['detection_scores'], category_index, max_boxes_to_draw=1, min_score_thresh=0)
                tf.keras.preprocessing.image.save_img(output_path+f'/img{i}.png', drawn_img[0])

            evaluator.dump(os.path.join(output_path, 'validation_results.pickle'))
            if comet:
                experiment.log_asset('validation_results.pickle')
            evaluator.calculate_default_and_save(output_path)

            extra_files.append(os.path.join(output_path, 'stats.json'))
            extra_files.append(os.path.join(output_path, 'validation_results.pickle'))
            extra_files += glob.glob(os.path.join(output_path, '*_curve_*.png'))
            if comet:
                experiment.log_asset(os.path.join(output_path, 'stats.json'))
                for img in glob.glob(os.path.join(output_path, '*_curve_*.png')):
                    experiment.log_image(img)

    #export detection_model as SavedModel
    saved_model_path = model_dir + '/export'
    configproto = config_util.create_pipeline_proto_from_configs(configs)
    export_inference_graph('image_tensor', configproto, model_dir, saved_model_path)

    if comet:
        experiment.log_asset_data(train.metadata, file_name="metadata.json")

    # export metadata locally
    with open(base_dir / 'metadata.json', 'w') as f:
        json.dump(train.metadata, f, indent=2)
        
    result = TrainOutput(Path(saved_model_path), extra_files)
    return result
    

### HELPERS ###

def load_fine_tune_checkpoint(
    model, checkpoint_path, checkpoint_type, checkpoint_version, input_dataset,
    unpad_groundtruth_tensors):

#NOTE: Moved this function from OD API to add 'expect_partial' to the restore
  """Load a fine tuning classification or detection checkpoint.
  To make sure the model variables are all built, this method first executes
  the model by computing a dummy loss. (Models might not have built their
  variables before their first execution)
  It then loads an object-based classification or detection checkpoint.
  This method updates the model in-place and does not return a value.
  Args:
    model: A DetectionModel (based on Keras) to load a fine-tuning
      checkpoint for.
    checkpoint_path: Directory with checkpoints file or path to checkpoint.
    checkpoint_type: Whether to restore from a full detection
      checkpoint (with compatible variable names) or to restore from a
      classification checkpoint for initialization prior to training.
      Valid values: `detection`, `classification`.
    checkpoint_version: train_pb2.CheckpointVersion.V1 or V2 enum indicating
      whether to load checkpoints in V1 style or V2 style.  In this binary
      we only support V2 style (object-based) checkpoints.
    input_dataset: The tf.data Dataset the model is being trained on. Needed
      to get the shapes for the dummy loss computation.
    unpad_groundtruth_tensors: A parameter passed to unstack_batch.
  Raises:
    IOError: if `checkpoint_path` does not point at a valid object-based
      checkpoint
    ValueError: if `checkpoint_version` is not train_pb2.CheckpointVersion.V2
  """
  if not model_lib2.is_object_based_checkpoint(checkpoint_path):
    raise IOError('Checkpoint is expected to be an object-based checkpoint.')
  if checkpoint_version == od.protos.train_pb2.CheckpointVersion.V1:
    raise ValueError('Checkpoint version should be V2')

  features, labels = iter(input_dataset).next()

  @tf.function
  def _dummy_computation_fn(features, labels):
    model._is_training = False  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(False)
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    return model_lib2._compute_losses_and_predictions_dicts(
        model,
        features,
        labels)

  strategy = tf.compat.v2.distribute.get_strategy()
  if hasattr(tf.distribute.Strategy, 'run'):
    strategy.run(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))
  else:
    strategy.experimental_run_v2(
        _dummy_computation_fn, args=(
            features,
            labels,
        ))

  restore_from_objects_dict = model.restore_from_objects(
      fine_tune_checkpoint_type=checkpoint_type)
  ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
  ckpt.restore(checkpoint_path).expect_partial()

def _get_paths_for_extra_files(artifact_path: Path):
    """Returns the filepaths for all checkpoint, config, and pbtxt (label)
    files in the artifact directory. Gets filepath for the exported inference
    graph.

    Args:
        artifact_path (Path): path to training artifacts
    
    Returns:
        list: list of Paths that point to files
    """
    extras = []
    # get checkpoints
    extras_path = artifact_path / 'models' / 'model'
    files = os.listdir(extras_path)

    # path to label map
    labels_path = artifact_path / 'data' / 'label_map.pbtxt'

    checkpoints = [f for f in files if checkpoint_regex.match(f)]

    # calculate the max checkpoint
    max_checkpoint = 0
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint.split('-')[1].split('.')[0])
        if checkpoint_num > max_checkpoint:
            max_checkpoint = checkpoint_num

    ckpt_prefix = 'model.ckpt-' + str(max_checkpoint)
    checkpoint_path = extras_path / ckpt_prefix
    pipeline_path = extras_path / 'pipeline.config'

    # append files to include in extras directory
    extras = [extras_path / f for f in checkpoints]
    extras.append(pipeline_path)
    #extras.append(extras_path / 'graph.pbtxt')

    # append event checkpoints for tensorboard
    for f in os.listdir(extras_path):
        if f.startswith('events.out'):
            extras.append(extras_path / f)

    #for f in os.listdir(extras_path / 'eval_0'):
    #    if f.startswith('events.out'):
    #        extras.append(extras_path / 'eval_0' / f)

    extras.append(labels_path)
    return extras, checkpoint_path

# stdout redirection found at https://codingdose.info/2018/03/22/supress-print-output-in-python/
def _import_od():
    """ Imports the necessary libraries for object detection training.
    Used to avoid importing them at the top of the file where they get imported
    on every ravenML command call, even those not to this plugin.
    
    Also suppresses warning outputs from the TF OD API.
    """
    # create a text trap and redirect stdout
    # to suppress printed warnings from object detection and tf
    text_trap = io.StringIO()
    sys.stdout = text_trap
    sys.stderr = text_trap
    
    # Calls to _dynamic_import below map to the following standard imports:
    #
    # import tensorflow as tf
    # from object_detection import model_hparams
    # from object_detection import model_lib
    _dynamic_import('tensorflow', 'tf')
    _dynamic_import('object_detection.model_hparams', 'model_hparams')
    _dynamic_import('object_detection.model_lib_v2', 'model_lib2')
    _dynamic_import('object_detection.model_lib', 'model_lib')
    _dynamic_import('object_detection.builders.model_builder', 'model_builder')
    _dynamic_import('object_detection.utils.config_util', 'config_util')
    #_dynamic_import('object_detection.exporter_lib_v2', 'exporter')
    _dynamic_import('object_detection.inputs', 'inputs')
    _dynamic_import('object_detection', 'od')
    _dynamic_import('object_detection.protos', 'pipeline_pb2', asfunction=True)
    
    # now restore stdout function
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
# this function is derived from https://stackoverflow.com/a/46878490
# NOTE: this function should be used in all plugins, but the function is NOT
# importable because of the use of globals(). You must copy the code.
def _dynamic_import(modulename, shortname = None, asfunction = False):
    """ Function to dynamically import python modules into the global scope.

    Args:
        modulename (str): name of the module to import (ex: os, ex: os.path)
        shortname (str, optional): desired shortname binding of the module (ex: import tensorflow as tf)
        asfunction (bool, optional): whether the shortname is a module function or not (ex: from time import time)
        
    Examples:
        Whole module import: i.e, replace "import tensorflow"
        >>> _dynamic_import('tensorflow')
        
        Named module import: i.e, replace "import tensorflow as tf"
        >>> _dynamic_import('tensorflow', 'tf')
        
        Submodule import: i.e, replace "from object_detction import model_lib"
        >>> _dynamic_import('object_detection.model_lib', 'model_lib')
        
        Function import: i.e, replace "from ravenml.utils.config import get_config"
        >>> _dynamic_import('ravenml.utils.config', 'get_config', asfunction=True)
        
    """
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = importlib.import_module(modulename)
    else:        
        globals()[shortname] = getattr(importlib.import_module(modulename), shortname)
