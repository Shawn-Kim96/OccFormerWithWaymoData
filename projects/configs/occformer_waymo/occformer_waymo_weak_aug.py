# Experiment: Weak Data Augmentation
# Compare with baseline to analyze augmentation impact
# Expected: Faster convergence, potential overfitting

_base_ = './occformer_waymo_baseline.py'

# Weak augmentation settings
bda_aug_conf = dict(
    rot_lim=(0, 0),             # No rotation
    scale_lim=(0.98, 1.02),     # Minimal scale (baseline: 0.95, 1.05)
    flip_dx_ratio=0.3,          # Less flipping (baseline: 0.5)
    flip_dy_ratio=0.3,
    flip_dz_ratio=0.3,
)

data_config = {
    'input_size': (640, 960),
    'resize': (-0.03, 0.05),    # Weaker resize (baseline: -0.06, 0.11)
    'rot': (-2.0, 2.0),         # Weaker rotation (baseline: -5.4, 5.4)
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
