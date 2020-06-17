"""
Author(s):      Nihal Dhamani (nihaldhamani@gmail.com), 
                Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/10/2019

Helper functions for the TF Bounding Box plugin.
"""

import os
import shutil
import tarfile
import yaml
import click
import urllib.request
from colorama import init, Fore
from pathlib import Path
from ravenml.utils.local_cache import LocalCache, global_cache
from ravenml.utils.question import user_confirms, user_input, user_selects

init()

# local cache for the tf_bbox plugin (within ravenML cache)
bbox_cache = LocalCache(global_cache.path / 'tf-bbox-tune')

def prepare_for_training(base_dir: Path, data_path: Path, arch_path: Path, model_type: str, metadata: dict):
    """ Prepares the system for training.

    Creates artifact directory structure. Prompts user for choice of optimizer and
    hyperparameters. Injects hyperparameters into config files. Adds hyperparameters
    to given metadata dictionary.

    Args:
        base_dir (Path): root of training directory
        data_path (Path): path to dataset
        arch_path (Path): path to model architecture directory
        model_type (str): name of model type (i.e, ssd_inception_v1)
        metadata (dict): metadata dictionary to add fields to

    Returns:
        bool: True if successful, False otherwise
    """
    # hyperparameter metadata dictionary
    hp_metadata = {}
    
    # check if base_dir exists already and prompt before overwriting
    if os.path.exists(base_dir):
        if user_confirms('Artifact storage location contains old data. Overwrite?'):
            shutil.rmtree(base_dir)
        else:
            return False
    os.makedirs(base_dir)
    
    # create a data folder within our base_directory
    os.makedirs(base_dir / 'data')

    # copy object-detection.pbtxt from dataset and move into training data folder
    pbtxt_file = data_path / 'label_map.pbtxt'
    shutil.copy(pbtxt_file, base_dir / 'data')

    # calculate number of classes from pbtxt file
    with open(pbtxt_file, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)
    
    # get num eval examples from file
    num_eval_file = data_path / 'splits/complete/train/test.record.numexamples'
    try:
        with open(num_eval_file, "r") as f:
            lines = f.readlines()
            num_eval_examples = int(lines[0])

    except:
        num_eval_examples = 1

    # create models, model, eval, and train folders
    model_folder = base_dir / 'models' / 'model'
    # model_folder = models_folder / 'model'
    # os.makedirs(models_folder)
    eval_folder = model_folder / 'eval'
    train_folder = model_folder / 'train'
    os.makedirs(model_folder)
    os.makedirs(eval_folder)
    os.makedirs(train_folder)
    
    # load optimizer choices and prompt for selection
    defaults = {}
    defaults_path = Path(os.path.dirname(__file__)) / 'model_defaults' / f'{model_type}_defaults.yml'
    with open(defaults_path, 'r') as stream:
        try:
            defaults = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    optimizer_name =  user_selects('Choose optimizer', defaults.keys())
    hp_metadata['optimizer'] = optimizer_name
    
    ### PIPELINE CONFIG CREATION ###
    # grab default config for the chosen optimizer
    default_config = defaults[optimizer_name]
    # prompt user for new configuration
    user_config = default_config
    # add to hyperparameter metadata dict
    for field, value in user_config.items():
        hp_metadata[field] = value
        
    # load template pipeline file
    cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    pipeline_file_name = f'{model_type}_{optimizer_name.lower()}.config'
    pipeline_path = cur_dir / 'pipeline_templates' / pipeline_file_name
    with open(pipeline_path) as template:
        pipeline_contents = template.read()
    
    # insert training directory path into config file
    if base_dir.name.endswith('/') or base_dir.name.endswith(r"\\"):
        pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir))
    else:
        if os.name == 'nt':
            pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir) + r"\\")
        else:
            pipeline_contents = pipeline_contents.replace('<replace_path>', str(base_dir) + '/')

    # place TF record files into training directory
    num_train_records = 0
    num_test_records = 0
    records_path = data_path / 'splits/complete/train'
    for record_file in os.listdir(records_path):
        if record_file.startswith('train.record-'):
            num_train_records += 1
            file_path = records_path / record_file
            shutil.copy(file_path, base_dir / 'data')

        if record_file.startswith('test.record-'):
            num_test_records += 1
            file_path = records_path / record_file
            shutil.copy(file_path, base_dir / 'data')

    # convert int to left zero padded string of length 5
    user_config['num_train_records'] = str(num_train_records).zfill(5)
    user_config['num_test_records'] = str(num_test_records).zfill(5)
            
    # insert rest of config into config file
    for key, value in user_config.items():
        formatted = '<replace_' + key + '>'
        pipeline_contents = pipeline_contents.replace(formatted, str(value))

    # insert num clases into config file
    pipeline_contents = pipeline_contents.replace('<replace_num_classes>', str(num_classes))

    # insert num eval examples into config file
    pipeline_contents = pipeline_contents.replace('<replace_num_eval_examples>', str(num_eval_examples))

    # output final configuation file for training
    with open(model_folder / 'pipeline.config', 'w') as file:
        file.write(pipeline_contents)

    # copy model checkpoints to our train folder
    checkpoint_folder = arch_path
    checkpoint0_folder = cur_dir / 'checkpoint_0'
    file1 = checkpoint_folder / 'model.ckpt.data-00000-of-00001'
    file2 = checkpoint_folder / 'model.ckpt.index'
    file3 = checkpoint_folder / 'model.ckpt.meta'
    file4 = checkpoint0_folder / 'model.ckpt-0.data-00000-of-00001'
    file5 = checkpoint0_folder / 'model.ckpt-0.index'
    file6 = checkpoint0_folder / 'model.ckpt-0.meta'
    shutil.copy2(file1, train_folder)
    shutil.copy2(file2, train_folder)
    shutil.copy2(file3, train_folder)
    shutil.copy2(file4, train_folder)
    shutil.copy2(file5, train_folder)
    shutil.copy2(file6, train_folder)
    
    # load starting checkpoint template and insert training directory path
    checkpoint_file = checkpoint0_folder / 'checkpoint'
    with open(checkpoint_file) as cf:
        checkpoint_contents = cf.read()
    checkpoint_contents = checkpoint_contents.replace('<replace>', str(train_folder))
    with open(train_folder / 'checkpoint', 'w') as new_cf:
        new_cf.write(checkpoint_contents)
    
    # update metadata and return success
    metadata['hyperparameters'] = hp_metadata
    return True

def download_model_arch(model_name):
    """Downloads the model architecture with the given name.

    Args:
        model_name (str): model type
    
    Returns:
        Path: path to model architecture
    """
    url = 'http://download.tensorflow.org/models/object_detection/%s.tar.gz' %(model_name)
    # make paths within bbox cache 
    bbox_cache.ensure_subpath_exists('bbox_model_archs')
    archs_path = bbox_cache.path / 'bbox_model_archs'
    untarred_path = archs_path / model_name
    # check if download is required
    if not bbox_cache.subpath_exists(untarred_path):
        click.echo("Model checkpoint not found in cache. Downloading...")
        # download tar file
        tar_name = url.split('/')[-1]
        tar_path = archs_path / tar_name
        urllib.request.urlretrieve(url, tar_path)
        
        click.echo("Untarring model checkpoint...")
        if (tar_name.endswith("tar.gz")):
            tar = tarfile.open(tar_path, "r:gz")
            tar.extractall(path=archs_path)
            tar.close()

        # get rid of tar file
        os.remove(tar_path)
    else:
        click.echo('Model checkpoint found in cache.')
        
    return untarred_path
    
def _configuration_prompt(current_config: dict):
    """Prompts user to allow editing of current training configuration.

    Args:
        current_config (dict): current training configuration
        
    Returns:
        dict: updated training configuration
    """
    _print_config(current_config)
    if user_confirms('Edit default configuration?'):
        for field in current_config:
            if user_confirms(f'Edit {field}? (default: {current_config[field]})'):
                current_config[field] = user_input(f'{field}:', default=str(current_config[field]))
    return current_config

def _print_config(config: dict):
    """Prints the given training configuration with colorization.

    Args:
        config (dict): training configuration to print
    """
    click.echo('Current configuration:')
    for field, value in config.items():
        click.echo(Fore.GREEN + f'{field}: ' + Fore.WHITE + f'{value}')
