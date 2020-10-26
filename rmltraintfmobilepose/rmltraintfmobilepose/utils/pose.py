from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import time


def solve_pose(
    ref_points,
    keypoints,
    focal_length,
    imdims,
    extra_crop_params=None,
    ransac=True,
    reduce_mean=False,
    return_inliers=False,
):
    """
    Calculates pose vectors using CV2's solvePNP.
    :param ref_points: 3D reference points, shape (n, 3)
    :param keypoints: 2D image keypoints in (y, x) pixel coordinates. Either shape (n, 2) or (m, n, 2), where n is
        the number of keypoints and m is the number of guesses per keypoint.
    :param focal_length: original camera focal length (vertical, horizontal)
    :param imdims: (height, width) current image dimensions
    :param extra_crop_params: a optional dict with extra parameters that are necessary to
        adjust for cropping and rescaling. If provided, must have the keys {'centroid', 'bbox_size', 'imdims'}
        where 'imdims' are the original image dimensions before cropping/rescaling.
    :param ransac: whether or not to use RANSAC on the guesses
    :param reduce_mean: if `keypoints` has shape (m, n, 2) and `reduce_mean` is true, then all the guesses for each
        keypoint will be reduced to one guess by their mean.
    """
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    imdims = np.array(imdims)
    if extra_crop_params:
        assert extra_crop_params.keys() == {"centroid", "bbox_size", "imdims"}
        original_imdims = np.array(extra_crop_params["imdims"])
        origin = (
            np.array(extra_crop_params["centroid"]) - extra_crop_params["bbox_size"] / 2
        )
        center = original_imdims / 2 - origin
        focal_length *= imdims / extra_crop_params["bbox_size"]
        center *= imdims / extra_crop_params["bbox_size"]
    else:
        center = imdims / 2

    cam_matrix = np.array(
        [[focal_length[1], 0, center[1]], [0, focal_length[0], center[0]], [0, 0, 1]],
        dtype=np.float32,
    )

    keypoints = keypoints[..., [1, 0]]
    if len(keypoints.shape) == 3:
        if reduce_mean:
            keypoints = keypoints.mean(axis=0)
        else:
            keypoints = keypoints.reshape([-1, 2])
            ref_points = np.tile(ref_points, [len(keypoints) // len(ref_points), 1])

    if not ransac:
        ret, r_vec, t_vec = cv2.solvePnP(
            ref_points, keypoints, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )
    else:
        # The reprojection error is the maximum pixel distance away
        # a keypoint can be to be considered an inlier. A higher value
        # can improve convergence speed in exchange for accuracy. The default
        # is 8.
        ret, r_vec, t_vec, inliers = cv2.solvePnPRansac(
            ref_points,
            keypoints,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            # reprojectionError=8
        )
    if not ret:
        print("Pose solve failed")
        r_vec, t_vec = np.zeros([2, 3], dtype=np.float32)
    if ransac and return_inliers:
        return r_vec, t_vec, len(inliers)
    return r_vec, t_vec


def to_rotation(r):
    if isinstance(r, Rotation):
        return r
    r = np.array(r).squeeze()
    if len(r) == 3:
        return Rotation.from_rotvec(r)
    if len(r) == 4:
        w, x, y, z = r
        return Rotation.from_quat([x, y, z, w])
    raise ValueError()


def geodesic_error(rot_pred, rot_truth, flip=False):
    rot_pred, rot_truth = to_rotation(rot_pred), to_rotation(rot_truth)
    # rot_pred = Rotation.from_euler("x", 180, degrees=True) * rot_pred
    err = _quat_geodesic_error(rot_pred.as_quat(), rot_truth.as_quat())
    if flip:
        rot_pred_flip = rot_pred * Rotation.from_euler("z", 180, degrees=True)
        return min(
            err, _quat_geodesic_error(rot_pred_flip.as_quat(), rot_truth.as_quat()),
        )
    return err


def _quat_geodesic_error(q1, q2):
    dot = np.abs(np.sum(q1 * q2))
    return 2 * np.arccos(np.clip(dot, 0, 1))


def position_error(pos_pred, pos_truth):
    pos_pred, pos_truth = np.array(pos_pred).squeeze(), np.array(pos_truth).squeeze()
    err = np.linalg.norm(pos_pred - pos_truth)
    return err, err / np.linalg.norm(pos_truth)


def display_geodesic_stats(errs_pose, errs_position):
    print(f"\n---- Geodesic Error Stats ----")
    stats = {
        "mean": np.mean(errs_pose),
        "median": np.median(errs_pose),
        "max": np.max(errs_pose),
    }
    for label, val in stats.items():
        print(f"{label:8s} = {val:.3f} ({np.degrees(val):.3f} deg)")
    if len(errs_position) == 0:
        return
    print(f"\n---- Position Error Stats ----")
    stats = {
        "mean": np.mean(errs_position),
        "median": np.median(errs_position),
        "max": np.max(errs_position),
    }
    for label, val in stats.items():
        print(f"{label:8s} = {val:.3f}")
    print(f"\n---- Combined Error Stats ----")
    stats = {
        "mean": np.mean(errs_position + errs_pose),
        "median": np.median(errs_position + errs_pose),
        "max": np.max(errs_position + errs_pose),
    }
    for label, val in stats.items():
        print(f"{label:8s} = {val:.3f}")


def display_keypoint_stats(errs):
    if len(errs) == 0:
        return
    errs = np.array(errs)
    print(f"\n---- Error Stats Per Keypoint ----")
    print(f" ### | mean | median | max ")
    for kp_idx in range(errs.shape[1]):
        err = errs[:, kp_idx]
        print(
            f" {kp_idx:<4d}| {np.mean(err):<5.2f}| {np.median(err):<7.2f}| {np.max(err):<4.2f}"
        )
