from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import user_confirms
from datetime import datetime
import tensorflow as tf
import numpy as np
import shutil
import click
import json
import cv2
import os

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import KeypointsModel
from . import utils


@click.group(help='TensorFlow Keypoints Regression.')
def tf_keypoints():
    pass


@tf_keypoints.command(help="Train a model.")
@pass_train
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.pass_context
def train(ctx, train: TrainInput, config):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train".
    # After training, create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = (LocalCache(global_cache.path / 'tf-keypoints').path if train.artifact_path is None \
        else train.artifact_path) / 'artifacts'

    if os.path.exists(artifact_dir):
        if user_confirms('Artifact storage location contains old data. Overwrite?'):
            shutil.rmtree(artifact_dir)
        else:
            return ctx.exit()
    os.makedirs(artifact_dir)

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"
    keypoints_path = train.dataset.path / "keypoints.npy"

    with open(config, "r") as f:
        hyperparameters = json.load(f)

    keypoints_3d = np.load(keypoints_path)

    # fill metadata
    metadata = {
        'architecture': 'keypoints_regression',
        'date_started_at': datetime.utcnow().isoformat() + "Z",
        'dataset_used': train.dataset.name,
        'config': hyperparameters
    }
    with open(artifact_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    trainer = KeypointsModel(data_dir, hyperparameters, keypoints_3d)
    model_path = trainer.train(artifact_dir)

    # get Tensorboard files
    # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    #   but ravenML trashes this structure and just uploads the individual files to S3.
    extra_files = []
    for dirpath, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    return TrainOutput(metadata, artifact_dir, model_path, extra_files, train.artifact_path is not None)


@tf_keypoints.command(help="Evaluate a model (Keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
@click.pass_context
def eval(ctx, train, model_path, pnp_crop_size=1024, pnp_focal_length=1422):
    model = tf.keras.models.load_model(model_path)
    cropsize = model.input.shape[1]
    nb_keypoints = model.output.shape[1] // 2
    ref_points = np.load(train.dataset.path / "keypoints.npy").reshape((-1, 3))[:nb_keypoints]
    test_data = utils.dataset_from_directory(train.dataset.path / "test", cropsize)
    test_data = test_data.map(
        lambda image, metadata: (
            tf.ensure_shape(image, [cropsize, cropsize, 3]),
            tf.ensure_shape(metadata["pose"], [4])
        )
    )
    test_data = test_data.batch(32)

    errs = []
    for image_batch, pose_batch in test_data.as_numpy_iterator():
        n = image_batch.shape[0]
        kps_pred = model.predict(image_batch)
        kps = ((kps_pred * (pnp_crop_size // 2)) + pnp_crop_size // 2).reshape((-1, nb_keypoints, 2))
        for i in range(n):
            r_vec, t_vec = utils.calculate_pose_vectors(
                ref_points, kps[i], 
                pnp_focal_length, pnp_crop_size)
            err = utils.rvec_geodesic_error(r_vec, pose_batch[i])
            errs.append(err)
    
    print('---- Geodesic Error Stats ----')
    stats = {
        'mean': np.mean(errs),
        'median': np.median(errs),
        'max': np.max(errs)
    }
    for label, val in stats.items():
        print('{:8s} = {:.3f} ({:.3f} deg)'.format(label, val, np.degrees(val)))

