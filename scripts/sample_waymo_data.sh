SUB=/home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1_subset/waymo_format
mkdir -p $SUB/training $SUB/validation $SUB/testing

ls /home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/waymo_format/training/*.tfrecord | head -n 80 | while read f; do ln -s "$f" $SUB/training/; done
ls /home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/waymo_format/validation/*.tfrecord | head -n 20 | while read f; do ln -s "$f" $SUB/validation/; done
ls /home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1/waymo_format/testing/*.tfrecord | head -n 20 | while read f; do ln -s "$f" $SUB/testing/; done

cd /home/018219422/mmdetection3d
PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=0 python tools/create_data.py waymo \
  --root-path /home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1_subset \
  --out-dir  /home/018219422/OccFormerWithWaymoData/data/waymo_v1-3-1_subset \
  --workers 1 \
  --extra-tag waymo \
  --version v1.4