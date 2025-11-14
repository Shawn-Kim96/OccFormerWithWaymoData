# Experiment: Reduced Number of Queries
# Compare with baseline (100 queries) to analyze efficiency vs accuracy tradeoff
# Expected: Faster inference, lower memory, potentially lower accuracy

_base_ = './occformer_waymo_baseline.py'

# Reduced queries for efficiency
mask2former_num_queries = 50  # Reduced from 100

model = dict(
    pts_bbox_head=dict(
        num_queries=mask2former_num_queries,
    ),
)

# Can potentially increase batch size with reduced queries
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
