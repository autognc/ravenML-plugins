import tensorflow as tf
import glob
import json
import os
import xml.etree.ElementTree as ET
from .train import PoseRegressionModel


def recursive_map_dict(d, f):
    if isinstance(d, dict):
        return {k: recursive_map_dict(v, f) for k, v in d.items()}
    return f(d)


def dataset_from_directory(dir_path, cropsize):
    """
    Get a Tensorflow dataset that generates samples from a directory with test data
    that is not in TFRecord format (i.e. a directory with image_*.png, meta_*.json, and bboxLabels_*.xml files).
    The images are cropped to the spacecraft using the bounding box truth data in the XML files.

    :param dir_path: the path to the directory
    :param cropsize: the output size for the images, in pixels

    :return: a Tensorflow dataset that generates (image, metadata) tuples where image is a [cropsize, cropsize, 3]
    Tensor and metadata is a nested dictionary of Tensors.
    """
    def generator():
        image_files = sorted(glob.glob(os.path.join(dir_path, "image_*")))
        meta_files = sorted(glob.glob(os.path.join(dir_path, "meta_*.json")))
        for image_file, meta_file in zip(image_files, meta_files):
            # load metadata
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            metadata = recursive_map_dict(metadata, tf.convert_to_tensor)

            # load bounding box
            bbox = metadata['bboxes']['cygnus']
            xmin = bbox['xmin']
            xmax = bbox['xmax']
            ymin = bbox['ymin']
            ymax = bbox['ymax']
            centroid = tf.convert_to_tensor([(ymax + ymin) / 2, (xmax + xmin) / 2], dtype=tf.float32)
            bbox_size = tf.cast(tf.maximum(xmax - xmin, ymax - ymin), tf.float32) * 1.25

            # load and crop image
            image_data = tf.io.read_file(image_file)
            image = PoseRegressionModel.preprocess_image(image_data, centroid, bbox_size, cropsize)
            yield image, metadata

    meta_file_0 = glob.glob(os.path.join(dir_path, "meta_*.json"))[0]
    with open(meta_file_0, "r") as f:
        meta0 = json.load(f)
    dtypes = recursive_map_dict(meta0, lambda x: tf.convert_to_tensor(x).dtype)
    return tf.data.Dataset.from_generator(generator, (tf.float32, dtypes))
