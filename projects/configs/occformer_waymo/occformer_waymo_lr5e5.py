# Experiment: Lower Learning Rate (5e-5)
# Compare with baseline (1e-4) to analyze learning rate impact
# Expected: More stable training, potentially better convergence but slower

_base_ = './occformer_waymo_baseline.py'

lr = 5e-5

optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
)

# Adjust learning rate schedule for lower LR
lr_config = dict(
    policy='step',
    step=[25, 28],  # Decay later than baseline
)
