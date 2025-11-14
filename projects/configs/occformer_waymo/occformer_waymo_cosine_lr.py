# Experiment: Cosine Learning Rate Schedule
# Compare with baseline (step decay) to analyze scheduler impact
# Expected: Smoother learning, potentially better final performance

_base_ = './occformer_waymo_baseline.py'

# Cosine learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
)

# Cosine works well with longer training
runner = dict(type='EpochBasedRunner', max_epochs=40)

evaluation = dict(
    interval=1,
    save_best='waymo_SSC_mIoU',
    rule='greater',
)
