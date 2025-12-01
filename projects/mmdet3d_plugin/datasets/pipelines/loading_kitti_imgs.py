# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES

import torch
from PIL import Image
from .loading_nusc_imgs import mmlabNormalize

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False, img_norm_cfg=None):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.img_norm_cfg = img_norm_cfg

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate

    def _get_cam2lidar(self, results, view_idx):
        if 'lidar2cam' in results:
            lidar2cam = torch.Tensor(results['lidar2cam'][view_idx])
            return lidar2cam.inverse()
        if 'sensor2ego' in results:
            return torch.Tensor(results['sensor2ego'][view_idx])
        raise KeyError('Cannot find lidar2cam or sensor2ego for view {}'.format(view_idx))

    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']
        if not isinstance(img_filenames, (list, tuple)):
            img_filenames = [img_filenames]

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        gt_depths = []
        sensor2sensors = []
        canvas = []

        for view_idx, img_path in enumerate(img_filenames):
            img_np = mmcv.imread(img_path, 'unchanged')
            if view_idx == 0:
                results['raw_img'] = img_np
            img = Image.fromarray(img_np)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip_flag, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip_flag,
                rotate=rotate)

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            intrin = torch.Tensor(results['cam_intrinsic'][view_idx])
            cam2lidar = self._get_cam2lidar(results, view_idx)
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img, img_norm_cfg=self.img_norm_cfg))
            rots.append(rot)
            trans.append(tran)
            intrins.append(intrin)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            gt_depths.append(torch.zeros(1))
            sensor2sensors.append(cam2lidar)

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        gt_depths = torch.stack(gt_depths)
        sensor2sensors = torch.stack(sensor2sensors)

        results['canvas'] = np.stack(canvas)

        return (imgs, rots, trans, intrins, post_rots, post_trans,
                gt_depths, sensor2sensors)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        
        return results
