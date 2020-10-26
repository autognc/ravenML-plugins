"""
Evaluate on a directory of test data.
"""

import tensorflow as tf
import numpy as np
import os
import click
import json
import cv2
import tqdm
from ravenml.utils.question import user_confirms
from .. import utils


def make_serializable(tensor):
    n = tensor.numpy()
    if isinstance(n, bytes):
        return n.decode("utf-8")
    return n.tolist()


@click.command(help="Evaluate a model (Keras .h5 format).")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--keypoints",
    help='Path to 3D reference points .npy file (optional). If omitted, looks for "{directory}/keypoints.npy"',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("-f", "--focal_length", type=float, required=True)
@click.option("-n", "--num", type=int, help="Number of images to process (optional)")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Path to write output JSON (optional)",
)
@click.option(
    "-r",
    "--render",
    type=click.Path(file_okay=False),
    help="Directory to store keypoint renders (optional)",
)
def main(model_path, directory, keypoints, focal_length, num, output, render):
    if output and os.path.exists(output):
        if not user_confirms("Output path exists. Overwrite?"):
            return

    if render:
        os.makedirs(render, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(), loss=lambda _: 0,
    )
    nb_keypoints = model.output.shape[-1] // 2
    cropsize = model.input.shape[1]

    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    all_ref_points = np.load(keypoints_path).reshape((-1, 3))
    ref_points = all_ref_points[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        os.path.join(directory, "test"), cropsize, nb_keypoints=nb_keypoints
    )
    if num:
        data = data.take(num)
    data = data.batch(32)

    results = []
    errs_pose = []
    errs_position = []
    errs_by_keypoint = []
    img_cnt = 0
    temp_data = []
    for image_batch, truth_batch in tqdm.tqdm(data):
        truth_batch = [
            dict(zip(truth_batch.keys(), t)) for t in zip(*truth_batch.values())
        ]
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch = kps_batch * (cropsize // 2) + (cropsize // 2)
        for encoded_image, kps, truth in zip(image_batch, kps_batch.numpy(), truth_batch):
            image = tf.cast(encoded_image * 127.5 + 127.5, tf.uint8).numpy()
            kps_true = (truth["keypoints"] - truth["centroid"]) / truth[
                "bbox_size"
            ] * cropsize + (cropsize // 2)
            r_vec, t_vec = utils.pose.solve_pose(
                ref_points,
                kps,
                [focal_length, focal_length],
                image.shape[:2],
                # extra_crop_params={
                #     "centroid": truth["centroid"],
                #     "bbox_size": truth["bbox_size"],
                #     "imdims": truth["imdims"],
                # },
                ransac=True,
                reduce_mean=False,
            )
            r_vec_true, t_vec_true = utils.pose.solve_pose(
                ref_points,
                kps_true.numpy(),
                [focal_length, focal_length],
                image.shape[:2],
                # extra_crop_params={
                #     "centroid": truth["centroid"],
                #     "bbox_size": truth["bbox_size"],
                #     "imdims": truth["imdims"],
                # },
                ransac=True,
                reduce_mean=False,
            )
            err_pose = utils.pose.geodesic_error(r_vec, truth["pose"])
            err_position = utils.pose.position_error(t_vec, truth["position"])[1]
            temp_data.append((encoded_image.numpy(), err_pose, err_position, all_ref_points, ref_points, kps, kps_true.numpy(), focal_length, r_vec, t_vec, r_vec_true, t_vec_true))
            errs_pose.append(err_pose)
            errs_position.append(err_position)
            # TODO doesn't use all guesses
            errs_by_keypoint.append(
                [np.linalg.norm(kp_true - kp) for kp, kp_true in zip(kps, kps_true)]
            )
            if render:
                kps = kps.reshape(-1, nb_keypoints, 2).transpose([1, 0, 2])
                hues = np.linspace(
                    0, 360, num=nb_keypoints, endpoint=False, dtype=np.float32
                )
                colors = np.stack(
                    [
                        hues,
                        np.ones(nb_keypoints, np.float32),
                        np.ones(nb_keypoints, np.float32),
                    ],
                    axis=-1,
                )
                colors = np.squeeze(cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR))
                colors = (colors * 255).astype(np.uint8)
                for color, guesses in zip(colors, kps):
                    for kp in guesses:
                        cv2.circle(
                            image, tuple(kp[::-1]), 3, tuple(map(int, color)), -1
                        )
                cv2.imwrite(os.path.join(render, f"{img_cnt:04d}.png"), image)
            result = utils.data.recursive_map_dict(truth, make_serializable)
            result.update(
                {
                    "detected_pose": utils.pose.to_rotation(r_vec)
                    .as_quat()[[3, 0, 1, 2]]
                    .tolist(),
                    "detected_position": np.squeeze(t_vec).tolist(),
                }
            )
            results.append(result)
            img_cnt += 1

    # np.save(f"{output_path}/pose_errs.npy", np.array(errs_pose))
    # np.save(f"{output_path}/position_errs.npy", np.array(errs_position))
    # np.save(f"{output_path}/keypoint_errs.npy", np.array(errs_by_keypoint))
    utils.pose.display_keypoint_stats(errs_by_keypoint)
    utils.pose.display_geodesic_stats(np.array(errs_pose), np.array(errs_position))

    np.save('error_data.npy', temp_data)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
