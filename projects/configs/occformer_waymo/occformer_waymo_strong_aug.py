# Experiment: Strong Data Augmentation
# Compare with baseline to analyze augmentation impact on generalization
# Expected: Better generalization, potentially slower convergence

_base_ = './occformer_waymo_baseline.py'

# Strong augmentation settings
bda_aug_conf = dict(
    rot_lim=(-10.0, 10.0),      # Stronger rotation (baseline: 0, 0)
    scale_lim=(0.85, 1.15),     # Stronger scale (baseline: 0.95, 1.05)
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,
)

data_config = {
    'input_size': (640, 960),
    'resize': (-0.15, 0.15),    # Stronger resize (baseline: -0.06, 0.11)
    'rot': (-8.0, 8.0),         # Stronger rotation (baseline: -5.4, 5.4)
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# May need more epochs for convergence with strong augmentation
runner = dict(type='EpochBasedRunner', max_epochs=35)
