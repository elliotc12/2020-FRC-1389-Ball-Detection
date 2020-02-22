#!/usr/bin/env bash

#this script will:
    # take raw supervisely tar.gz
    # untar it, convert to the data/train data/test structure
    # convert JSON to CSV, then CSV to .record
    # create .pbtxt file in training/sdfsd (or data not sure)
    # create training/ folder
    # 

# Setup data directories
# tar -xf 'Images Tagged as Valid'.tar
# mv 'Images Tagged as Valid' data

# This whole exercise works with Python 3.6.5

# Following this guide:
# https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

# First, download images/json from supervisely
# Also, download any models you like from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# Gcloud stuff
export CLOUDSDK_PYTHON=`which python`
export PROJECT="flawless-earth-267313"
export GCS_BUCKET="elc-training-bucket-2020"
export TPU_ACCOUNT="service-772840404646@cloud-tpu.iam.gserviceaccount.com"
gcloud config set project $PROJECT
# Uncomment next line if making new bucket:
# gsutil mb gs://$GCS_BUCKET
# gcloud projects add-iam-policy-binding $PROJECT  --member serviceAccount:$TPU_ACCOUNT --role roles/ml.serviceAgent

# Generating .record files
# mkdir -p training
# python3 json_to_csv.py
# python3 generate_tfrecord.py --csv_input=training/train.csv  --output_path=training/train.record
# python3 generate_tfrecord.py --csv_input=training/eval.csv   --output_path=training/test.record

# Uploading to Gcloud
# gsutil -m cp -r training/*.record gs://${GCS_BUCKET}/data/
# gsutil cp label_map.pbtxt gs://${GCS_BUCKET}/data/label_map.pbtxt

# gsutil cp models/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt.* gs://${GCS_BUCKET}/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/
# gsutil cp models/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/model.ckpt.* gs://${GCS_BUCKET}/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/
# gsutil cp models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/model.ckpt.* gs://${GCS_BUCKET}/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/

# gsutil cp configs/faster_rcnn_inception_resnet_v2_atrous_coco.config gs://${GCS_BUCKET}/data/faster_rcnn_inception_resnet_v2_atrous_coco.config
# gsutil cp configs/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync.config gs://${GCS_BUCKET}/data/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync.config
gsutil cp configs/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config gs://${GCS_BUCKET}/data/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config
# gsutil cp configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config gs://${GCS_BUCKET}/data/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config

cd ~/models/research/
# bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
# python setup.py sdist
# (cd slim && python setup.py sdist)

gcloud ai-platform jobs submit training `whoami`_object_detection_`date +%s` \
       --job-dir=gs://${GCS_BUCKET}/train/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync/ \
       --packages ~/models/research/dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
       --module-name object_detection.model_tpu_main \
       --runtime-version 1.15 \
       --scale-tier BASIC_TPU \
       --region us-central1 \
       -- \
       --model_dir=gs://${GCS_BUCKET}/train/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync/ \
       --tpu_zone us-central1 \
       --pipeline_config_path=gs://${GCS_BUCKET}/data/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config

gcloud ai-platform jobs submit training `whoami`_object_detection_eval_validation_`date +%s` \
       --job-dir=gs://${GCS_BUCKET}/train/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync \
       --packages ~/models/research/dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
       --module-name object_detection.model_main \
       --runtime-version 1.15 \
       --scale-tier BASIC_GPU \
       --region us-central1 \
       -- \
       --model_dir=gs://${GCS_BUCKET}/train/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync/ \
       --pipeline_config_path=gs://${GCS_BUCKET}/data/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync.config \
       --checkpoint_dir=gs://${GCS_BUCKET}/train/ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync/

# play with model, batch size, loss function to change model performance
# maybe TURN OFF quantization??!?!?! it decreases precision by decreasing the size of weights
