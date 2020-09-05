import tqdm
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import user_confirms
from ravenml.utils.dataset import get_dataset, dataset_cache
from ravenml.data.interfaces import Dataset
from ravenml.utils.plugins import raise_parameter_error
from datetime import datetime
import matplotlib.pyplot as plt
from comet_ml import Experiment
from contextlib import ExitStack
import tensorflow as tf
import numpy as np
import random
import shutil
import click
import json
import glob
import os
import cv2
from scipy.spatial.transform import Rotation

from .train import KeypointsModel, PoseErrorCallback
from . import utils, data_utils


@click.group(help='TensorFlow Keypoints Regression.')
def tf_keypoints():
    pass


@tf_keypoints.command(help="Train a model.")
@pass_train
@click.option("--comet", type=str, help="Enable comet integration under an experiment by this name", default=None)
@click.pass_context
def train(ctx, train: TrainInput, comet):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # object creation, after which execution will fail as this means
    # the user did not pass a config. see ravenml core file train/commands.py for more detail

    # NOTE: after training, you must create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"
    keypoints_path = train.dataset.path / "keypoints.npy"

    hyperparameters = train.plugin_config

    keypoints_3d = np.load(keypoints_path)

    # fill metadata
    train.plugin_metadata['architecture'] = 'keypoints_regression'
    train.plugin_metadata['config'] = hyperparameters

    experiment = None
    if comet:
        experiment = Experiment(workspace='seeker-rd', project_name='keypoints-pose-regression')
        experiment.set_name(comet)
        experiment.log_parameters(hyperparameters)
        experiment.set_os_packages()
        experiment.set_pip_packages()

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    trainer = KeypointsModel(data_dir, hyperparameters, keypoints_3d)
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.train())
        model_path = trainer.train(artifact_dir, experiment)
    if experiment:
        experiment.end()

    # get Tensorboard files
    # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    #   but ravenML trashes this structure and just uploads the individual files to S3.
    extra_files = []
    for dirpath, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    return TrainOutput(model_path, extra_files)


@tf_keypoints.command(help="Evaluate a model (Keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('dataset_name', type=str)
@click.argument('output_path', type=click.Path(exists=False))
@click.option('--pnp_focal_length', default=1422.0)
@click.option('--plot', is_flag=True)
@click.option('--render_poses', is_flag=True)
@click.pass_context
def eval(ctx, model_path, dataset_name, output_path, pnp_focal_length, plot=False, render_poses=False):
    RAVEN_VAR = os.environ['RAVEN_VAR']
    print('ENV = ', RAVEN_VAR)
    # ensure dataset exists and get its path
    if os.path.exists(dataset_cache.path / dataset_name):
        dataset = Dataset(dataset_name, {}, dataset_cache.path / dataset_name)
    else:
        try:
            dataset = get_dataset(dataset_name)
        # exit if the dataset could not be found on S3
        except ValueError as e:
            raise_parameter_error(dataset_name, 'dataset name')

    if os.path.exists(output_path):
        if True or user_confirms('Artifact storage location contains old data. Overwrite?'):
            shutil.rmtree(output_path)
        else:
            return ctx.exit()
    os.makedirs(output_path)

    errs_pose = []
    errs_position = []
    errs_by_keypoint = []
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.name == 'mobilepose':
        nb_keypoints = model.output.shape[-1] // 2
    else:
        nb_keypoints = model.output.shape[1] // 2
    cropsize = model.input.shape[1]
    ref_points = np.load(dataset.path / "keypoints.npy").reshape((-1, 3))[:nb_keypoints]

    pose_error_callback = PoseErrorCallback(ref_points, cropsize, pnp_focal_length)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=KeypointsModel.make_mse_loss(keypoints_mode='coords'), # TODO check if mask
        metrics=[pose_error_callback.assign_metric]
    )

    test_data = data_utils.dataset_from_directory(dataset.path / "test", cropsize, nb_keypoints)
    test_data = test_data.batch(32)
    img_cnt = 0
    from collections import defaultdict
    examples = []
    for image_batch, truth_batch in tqdm.tqdm(test_data):
        kps_batch = model.predict(image_batch)
        if model.name == 'mobilepose':
            kps_batch = KeypointsModel.decode_displacement_field(kps_batch)
            kps_batch = tf.transpose(kps_batch, [0, 3, 1, 2])
            # if using the reduce_mean strategy, comment out the next line
            # and pass ransac=False to calculate_pose_vectors.
            kps_batch = tf.reshape(kps_batch, [tf.shape(kps_batch)[0], -1, 2]).numpy()
            # kps_batch = tf.reduce_mean(kps_batch, axis=1).numpy()
        else:
            kps_batch = kps_batch.reshape(kps_batch.shape[0], -1, 2)
        kps_batch = kps_batch * (cropsize // 2) + (cropsize // 2)
        kps_true_batch = (truth_batch['keypoints'] - truth_batch['centroid'][:, None, :])\
            / truth_batch['bbox_size'][:, None, None] * cropsize + (cropsize // 2)
        for i, (kps, kps_true) in enumerate(zip(kps_batch, kps_true_batch.numpy())):
            image = ((image_batch[i].numpy() + 1) / 2 * 255).astype(np.uint8)
            # print(kps, nb_keypoints, kps.shape)
            # 11 (2156, 2)
            a = kps.reshape((11, 2, 14 * 14))
            b = np.mean(a, axis=2).reshape((11, 2, 1))
            metrics = {}
            metrics['kps_grid_diff_mean'] = (np.mean(np.abs(a - b)))
            metrics['kps_grid_diff_max'] = (np.max(np.abs(a - b)))
            metrics['kps_grid_diff_min'] = (np.min(np.abs(a - b)))
            metrics['kps_grid_diff_std'] = (np.std(np.abs(a - b)))
            r_vec, t_vec, cam_matrix, coefs, inliers = utils.calculate_pose_vectors(
                ref_points, kps,
                [pnp_focal_length, pnp_focal_length], image.shape[:2],
                extra_crop_params={
                    'centroid': truth_batch['centroid'][i],
                    'bbox_size': truth_batch['bbox_size'][i],
                    'imdims': truth_batch['imdims'][i],
                },
                ransac=True,
            )
            err = utils.geodesic_error(r_vec, truth_batch['pose'][i])
            examples.append((a, inliers, err))
            errs_pose.append(err)
            errs_position.append(
                np.linalg.norm(truth_batch['position'][i] - np.squeeze(t_vec)) / np.linalg.norm(truth_batch['position'][i])
            )
            # TODO doesn't use all guesses for mobilepose
            errs_by_keypoint.append([
                np.linalg.norm(kp_true - kp)
                for kp, kp_true in zip(kps, kps_true)
            ])
            if render_poses:
                kps = kps.reshape(-1, nb_keypoints, 2).transpose([1, 0, 2])
                hues = np.linspace(0, 360, num=nb_keypoints, endpoint=False, dtype=np.float32)
                colors = np.stack([hues, np.ones(nb_keypoints, np.float32), np.ones(nb_keypoints, np.float32)],
                                  axis=-1)
                colors = np.squeeze(cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR))
                colors = (colors * 255).astype(np.uint8)
                for color, guesses in zip(colors, kps):
                    for kp in guesses:
                        cv2.circle(image, tuple(kp[::-1]), 3, tuple(map(int, color)), -1)
                cv2.imwrite(f'{output_path}/pose-render-{img_cnt:04d}.png', image)
            img_cnt += 1
            if render_poses:
                break
        if render_poses:
            break



    np.save(f'{output_path}/pose_errs.npy', np.array(errs_pose))
    np.save(f'{output_path}/position_errs.npy', np.array(errs_position))
    np.save(f'{output_path}/keypoint_errs.npy', np.array(errs_by_keypoint))
    _display_keypoint_stats(errs_by_keypoint)
    display_geodesic_stats('Model Preds', np.array(errs_pose), np.array(errs_position), plot=plot)

    err_data = {
        'position_err_mean': np.mean(errs_position),
        'position_err_median': np.median(errs_position),
        'position_err_max': np.max(errs_position),
        'pose_err_mean': np.degrees(np.mean(errs_pose)),
        'pose_err_median': np.degrees(np.median(errs_pose)),
        'pose_err_max': np.degrees(np.max(errs_pose)),
    }

    np.save('err-data-{}.npy'.format(RAVEN_VAR), examples)


@tf_keypoints.command(help="Evaluate ground truth PnP.")
@click.argument('dataset_name', type=str)
@click.option('--keypoints', default=20)
@click.option('--pnp_focal_length', default=1422.0)
@click.option('--swap_random_percent', default=0, help="Randomly swap keypoints to test pnp.")
@click.pass_context
def evalpnp(ctx, dataset_name, keypoints, pnp_focal_length, swap_random_percent):
    # ensure dataset exists and get its path
    try:
        dataset = get_dataset(dataset_name)
    # exit if the dataset could not be found on S3
    except ValueError as e:
        raise_parameter_error(dataset_name, 'dataset name')

    nb_keypoints = keypoints
    errs_pose = []
    errs_position = [0]

    rand_swap_amt = int(swap_random_percent / 100 * nb_keypoints)
    if rand_swap_amt > 0:
        print('WARN: Randomly swapping {} keypoints.'.format(rand_swap_amt))

    ref_points = np.load(dataset.path / "keypoints.npy").reshape((-1, 3))
    meta_files = sorted(glob.glob(str(dataset.path / 'test' / "meta_*.json")))
    for meta_file in tqdm.tqdm(meta_files):
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        # FIXME: don't hardcode image resolution
        kps = np.array(metadata['keypoints'], np.float32) * 1024
        pose = metadata['pose']

        if rand_swap_amt > 0:
            swaps = random.sample(list(range(nb_keypoints)), k=rand_swap_amt * 2)
            for a, b in zip(swaps[::2], swaps[1::2]):
                kps[a], kps[b] = kps[b], kps[a]

        # FIXME: don't hardcode image resolution
        r_vec, t_vec, cam_matrix, coefs = utils.calculate_pose_vectors(
            ref_points[:nb_keypoints], kps[:nb_keypoints],
            [pnp_focal_length, pnp_focal_length], [1024, 1024])
        errs_pose.append(utils.geodesic_error(r_vec, pose))
        # position = np.array(metadata['position'])
        # errs_position.append(
            # np.linalg.norm(position - np.squeeze(t_vec)) / np.linalg.norm(position)
        # )

    display_geodesic_stats('PnP on truth, |kps|={})'.format(nb_keypoints), errs_pose, errs_position)


def display_geodesic_stats(title, errs_pose, errs_position, plot=False):
    print(f'\n---- Geodesic Error Stats ({title}) ----')
    stats = {
        'mean': np.mean(errs_pose),
        'median': np.median(errs_pose),
        'max': np.max(errs_pose)
    }
    for label, val in stats.items():
        print(f'{label:8s} = {val:.3f} ({np.degrees(val):.3f} deg)')
    print(f'\n---- Position Error Stats ({title}) ----')
    stats = {
        'mean': np.mean(errs_position),
        'median': np.median(errs_position),
        'max': np.max(errs_position)
    }
    for label, val in stats.items():
        print(f'{label:8s} = {val:.3f}')
    print(f'\n---- Combined Error Stats ({title}) ----')
    stats = {
        'mean': np.mean(errs_position + errs_pose),
        'median': np.median(errs_position + errs_pose),
        'max': np.max(errs_position + errs_pose)
    }
    for label, val in stats.items():
        print(f'{label:8s} = {val:.3f}')
    if plot:
        plt.hist([np.degrees(val) for val in errs_pose])
        plt.title(title)
        plt.show()


def _display_keypoint_stats(errs):
    errs = np.array(errs)
    print(f'\n---- Error Stats Per Keypoint ----')
    print(f' ### | mean | median | max ')
    for kp_idx in range(errs.shape[1]):
        err = errs[:, kp_idx]
        print(f' {kp_idx:<4d}| {np.mean(err):<5.2f}| {np.median(err):<7.2f}| {np.max(err):<4.2f}')
