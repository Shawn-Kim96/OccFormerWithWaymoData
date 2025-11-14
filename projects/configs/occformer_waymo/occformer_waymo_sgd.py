# Experiment: SGD Optimizer with Momentum
# Compare with baseline (AdamW) to analyze optimizer impact
# Expected: Different convergence pattern, may need careful tuning

_base_ = './occformer_waymo_baseline.py'

# SGD optimizer configuration
optimizer = dict(
    type='SGD',
    lr=0.01,  # SGD typically needs higher learning rate
    momentum=0.9,
    weight_decay=0.0001,
)

# SGD often benefits from cosine schedule
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
)
