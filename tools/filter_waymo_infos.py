#!/usr/bin/env python
"""Filter Waymo KITTI-format metadata to match existing files.

This script scans the KITTI-format Waymo directory, drops samples whose
images/LiDAR/occupancy labels are missing, optionally downsamples the
remainder, and rewrites both the info PKL and the ImageSets split file.

Example:
    python tools/filter_waymo_infos.py \
        --data-root data/waymo_v1-3-1/kitti_format \
        --ann-file data/waymo_v1-3-1/occ3d_waymo/waymo_infos_train.pkl \
        --out-ann-file data/waymo_v1-3-1/occ3d_waymo/waymo_infos_train.filtered.pkl \
        --split train --write-imageset --keep-fraction 0.1 --check-occ
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import mmcv


VIEW_SWAP = {2: 3, 3: 2}
OCC_KEYS = [
    'gt_occ', 'occ_path', 'occ_gt_path', 'occupancy_path',
    'occ_label_path', 'voxel_label_path'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Filter Waymo info PKLs')
    parser.add_argument('--data-root', required=True,
                        help='Path to data/waymo_v1-3-1/kitti_format')
    parser.add_argument('--ann-file', required=True,
                        help='Input info PKL to filter')
    parser.add_argument('--out-ann-file', required=True,
                        help='Output path for filtered info PKL')
    parser.add_argument('--split', default='train',
                        help='Split name (used when rewriting ImageSets)')
    parser.add_argument('--write-imageset', action='store_true',
                        help='Rewrite ImageSets/<split>.txt with kept ids')
    parser.add_argument('--keep-fraction', type=float, default=1.0,
                        help='Optional fraction of samples to retain (0-1].')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for subsampling')
    parser.add_argument('--num-views', type=int, default=5,
                        help='Number of camera views to verify.')
    parser.add_argument('--check-occ', action='store_true',
                        help='Verify occupancy labels referenced in the info')
    parser.add_argument('--occ-root', default=None,
                        help='Override root for occupancy paths. Defaults to '
                             'directory of ann-file.')
    return parser.parse_args()


def _expand_view_paths(base_image: Path, num_views: int) -> Tuple[Path, ...]:
    paths = []
    base_str = str(base_image)
    if 'image_0' not in base_str:
        return (base_image,)
    for idx in range(num_views):
        mapped = VIEW_SWAP.get(idx, idx)
        paths.append(Path(base_str.replace('image_0', f'image_{mapped}')))
    return tuple(paths)


def _resolve_occ_path(value: str, data_root: Path, occ_root: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    cand_paths = []
    path_obj = Path(value)
    if path_obj.is_absolute():
        cand_paths.append(path_obj)
    else:
        cand_paths.append(data_root / value)
        if occ_root is not None:
            cand_paths.append(occ_root / value)
        cand_paths.append(Path(value))
    for cand in cand_paths:
        if cand.exists():
            return cand
    return cand_paths[0]


def _resolve_occ_value(info: dict, data_root: Path, occ_root: Optional[Path]) -> Optional[Path]:
    for key in OCC_KEYS:
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return _resolve_occ_path(value, data_root, occ_root)
        if isinstance(value, dict):
            for subkey in ('path', 'file', 'data'):
                if subkey in value and isinstance(value[subkey], str):
                    return _resolve_occ_path(value[subkey], data_root, occ_root)
        if isinstance(value, (list, tuple)) and value:
            for item in value:
                if isinstance(item, str):
                    return _resolve_occ_path(item, data_root, occ_root)
    return None


def validate_sample(info: dict,
                    data_root: Path,
                    num_views: int,
                    check_occ: bool,
                    occ_root: Optional[Path]) -> Tuple[bool, Optional[str]]:
    sample_idx = info['image']['image_idx']
    base_img = data_root / info['image']['image_path']
    if not base_img.exists():
        return False, f'image_0 missing ({sample_idx})'

    for view_path in _expand_view_paths(base_img, num_views):
        if not view_path.exists():
            return False, f'multi-view missing ({sample_idx})'

    velo_path = data_root / info['point_cloud']['velodyne_path']
    if not velo_path.exists():
        return False, f'velodyne missing ({sample_idx})'

    if check_occ:
        occ_path = _resolve_occ_value(info, data_root, occ_root)
        if occ_path is None or not occ_path.exists():
            return False, f'occ missing ({sample_idx})'

    return True, None


def maybe_subsample(infos, keep_fraction: float, seed: int):
    if not (0 < keep_fraction < 1.0):
        return infos
    count = max(1, int(len(infos) * keep_fraction))
    random.Random(seed).shuffle(infos)
    return infos[:count]


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    occ_root = Path(args.occ_root) if args.occ_root else Path(args.ann_file).parent

    infos = mmcv.load(args.ann_file)
    kept = []
    stats = defaultdict(int)

    for info in infos:
        ok, reason = validate_sample(
            info,
            data_root=data_root,
            num_views=args.num_views,
            check_occ=args.check_occ,
            occ_root=occ_root)
        if ok:
            kept.append(info)
        else:
            stats[reason] += 1

    if not kept:
        raise RuntimeError('No samples left after filtering. Check your paths.')

    kept = maybe_subsample(kept, args.keep_fraction, args.seed)

    mmcv.dump(kept, args.out_ann_file)
    print(f'Saved {len(kept)} samples to {args.out_ann_file}')

    if args.write_imageset:
        imageset_dir = data_root / 'ImageSets'
        imageset_dir.mkdir(parents=True, exist_ok=True)
        split_file = imageset_dir / f'{args.split}.txt'
        with open(split_file, 'w') as f:
            for info in sorted(kept, key=lambda x: x['image']['image_idx']):
                f.write(f"{info['image']['image_idx']:07d}\n")
        print(f'Updated ImageSets file: {split_file}')

    if stats:
        print('Summary of removed samples:')
        for reason, count in stats.items():
            print(f'  {reason}: {count}')


if __name__ == '__main__':
    main()
