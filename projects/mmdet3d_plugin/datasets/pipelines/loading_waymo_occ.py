"""
Waymo-specific occupancy loading with automatic GT resizing.
This solves the grid size mismatch (200,200,16) -> (256,256,32) issue.
"""
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from .loading_nusc_occ import custom_rotate_3d
from scipy.ndimage import zoom


@PIPELINES.register_module()
class LoadWaymoOccAnnotation():
    """Load Waymo occupancy annotations with automatic resizing.

    Args:
        bda_aug_conf (dict): BDA augmentation config
        is_train (bool): Training mode flag
        point_cloud_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        target_occ_size (list): Target occupancy grid size [X, Y, Z]. Default: None (use GT size)
        resize_method (str): 'nearest' or 'trilinear'. Default: 'nearest' (better for labels)
    """
    def __init__(self,
                 bda_aug_conf,
                 is_train=True,
                 point_cloud_range=[0, -40, -3, 70.4, 40, 4],
                 target_occ_size=None,
                 resize_method='nearest'):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2
        self.target_occ_size = target_occ_size  # e.g., [256, 256, 32] or None
        self.resize_method = resize_method

        if self.target_occ_size is not None:
            print(f"[LoadWaymoOccAnnotation] Will resize GT from (200,200,16) to {tuple(self.target_occ_size)}")

    def resize_occupancy(self, occ_labels, target_size):
        """Resize occupancy labels to target size.

        Args:
            occ_labels (np.ndarray): Input occupancy labels [H, W, D]
            target_size (list): Target size [H', W', D']

        Returns:
            np.ndarray: Resized occupancy labels
        """
        if occ_labels.shape == tuple(target_size):
            return occ_labels

        # Calculate zoom factors
        zoom_factors = [
            target_size[0] / occ_labels.shape[0],
            target_size[1] / occ_labels.shape[1],
            target_size[2] / occ_labels.shape[2],
        ]

        if self.resize_method == 'nearest':
            # Nearest neighbor - best for discrete labels
            resized = zoom(occ_labels, zoom_factors, order=0, mode='nearest')
        else:
            # Trilinear interpolation (not recommended for labels, but provided as option)
            resized = zoom(occ_labels, zoom_factors, order=1, mode='nearest')
            resized = np.round(resized).astype(occ_labels.dtype)

        return resized

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf['flip_dz_ratio']

        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def forward_test(self, results):
        bda_rot = torch.eye(4).float()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)

        return results

    def __call__(self, results):
        if results.get('gt_occ', None) is None:
            return self.forward_test(results)

        # Convert to numpy for processing
        if type(results['gt_occ']) is list:
            gt_occ = [np.array(x) for x in results['gt_occ']]
            is_list = True
        else:
            gt_occ = np.array(results['gt_occ'])
            is_list = False

        # Process single array or list
        def process_single(occ):
            # Step 1: Remap labels (Waymo free space: 23 -> 15)
            occ = occ.copy()
            occ[occ == 23] = 15

            # Step 2: Resize if target size is specified
            if self.target_occ_size is not None:
                occ = self.resize_occupancy(occ, self.target_occ_size)

            return torch.from_numpy(occ).long()

        if is_list:
            gt_occ = [process_single(x) for x in gt_occ]
        else:
            gt_occ = process_single(gt_occ)

        # Apply BDA augmentation
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_bda_augmentation()
            gt_occ, bda_rot = voxel_transform(gt_occ, rotate_bda, scale_bda,
                        flip_dx, flip_dy, flip_dz, self.transform_center)
        else:
            bda_rot = torch.eye(4).float()

        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
        results['gt_occ'] = gt_occ.long()

        return results


def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz, transform_center=None):
    """Apply geometric transformations to voxel labels."""
    assert transform_center is not None
    trans_norm = torch.eye(4)
    trans_norm[:3, -1] = - transform_center
    trans_denorm = torch.eye(4)
    trans_denorm[:3, -1] = transform_center

    # BEV rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    # Flip matrices
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]])

    # Combined transform
    bda_mat = trans_denorm @ flip_mat @ rot_mat @ trans_norm

    # Apply transforms to voxel labels
    voxel_labels = voxel_labels.numpy().astype(np.uint8)

    if not np.isclose(rotate_degree, 0):
        voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)

    if flip_dz:
        voxel_labels = voxel_labels[:, :, ::-1]

    if flip_dy:
        voxel_labels = voxel_labels[:, ::-1]

    if flip_dx:
        voxel_labels = voxel_labels[::-1]

    voxel_labels = torch.from_numpy(voxel_labels.copy()).long()

    return voxel_labels, bda_mat
