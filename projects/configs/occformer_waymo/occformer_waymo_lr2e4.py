# Experiment: Higher Learning Rate (2e-4)
# Compare with baseline (1e-4) to analyze learning rate impact
# Expected: Faster initial learning, potential instability

_base_ = './occformer_waymo_baseline.py'

lr = 2e-4

optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
)

# Adjust learning rate schedule for higher LR - decay earlier
lr_config = dict(
    policy='step',
    step=[15, 22],  # Decay earlier than baseline
)
