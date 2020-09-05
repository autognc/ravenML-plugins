from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import time


def calculate_pose_vectors(ref_points, keypoints, focal_length, imdims, extra_crop_params=None, ransac=True):
    """
    Calculates pose vectors using CV2's solvePNP.
    :param ref_points: 3D reference points, shape (n, 3)
    :param keypoints: 2D image keypoints in (y, x) pixel coordinates. Shape (m, 2), where m>=n and m is a
        multiple of n. If m is greater than n, then keypoints will be interpreted as m // n guesses at the
        location of each 3D keypoint. These guesses should be in guess-major order: i.e. (num_guesses, n).reshape(-1).
    :param focal_length: original camera focal length (vertical, horizontal)
    :param imdims: (height, width) current image dimensions
    :param extra_crop_params: a optional dict with extra parameters that are necessary to
        adjust for cropping and rescaling. If provided, must have the keys {'centroid', 'bbox_size', 'imdims'}
        where 'imdims' are the original image dimensions before cropping/rescaling.
    """
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    imdims = np.array(imdims)
    if extra_crop_params:
        assert extra_crop_params.keys() == {'centroid', 'bbox_size', 'imdims'}
        original_imdims = np.array(extra_crop_params['imdims'])
        origin = np.array(extra_crop_params['centroid']) - extra_crop_params['bbox_size'] / 2
        center = original_imdims / 2 - origin
        focal_length *= imdims / extra_crop_params['bbox_size']
        center *= imdims / extra_crop_params['bbox_size']
    else:
        center = imdims / 2

    cam_matrix = np.array([
        [focal_length[1], 0, center[1]],
        [0, focal_length[0], center[0]],
        [0, 0, 1]
    ], dtype=np.float32)

    assert len(keypoints) % len(ref_points) == 0
    if len(keypoints) > len(ref_points):
        ref_points = np.tile(ref_points, [len(keypoints) // len(ref_points), 1])

    keypoints = keypoints[:, [1, 0]]
    if not ransac:
        ret, r_vec, t_vec = cv2.solvePnP(
            ref_points, keypoints,
            cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    else:
        # The reprojection error is the maximum pixel distance away
        # a keypoint can be to be considered an inlier. A higher value
        # can improve convergence speed is exchange for accuracy. The default
        # is 8.
        ret, r_vec, t_vec, inliers = cv2.solvePnPRansac(
            ref_points, keypoints,
            cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP,
            # reprojectionError=8
        )
    if not ret:
        print('Pose solve failed')
        r_vec, t_vec = np.zeros([2, 3], dtype=np.float32)
    return r_vec, t_vec, cam_matrix, dist_coeffs, inliers


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


def geodesic_error(rot_pred, rot_truth):
    rot_pred, rot_truth = to_rotation(rot_pred), to_rotation(rot_truth)
    # fix weird 180-degree flip
    # rot_pred = Rotation.from_euler('xyz', [np.pi, 0, 0]) * rot_pred
    return _quat_geodesic_error(rot_pred.as_quat(), rot_truth.as_quat())


def _quat_geodesic_error(q1, q2):
    dot = np.abs(np.sum(q1 * q2))
    return 2 * np.arccos(np.clip(dot, 0, 1))


def pow2_round(num):
    return int(2**round(np.log2(num)))
