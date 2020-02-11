#!/usr/bin/env bash
# python3 detectron2/demo/demo.py \
# --config-file ./detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml \
# --video-input source/mondogrosso_short.mp4 \
# --output /Volumes/hqueue/deep/StyleDance/output/detectron2 \
# --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
python3 make_data.py source/chemicalbros.mp4 output/chemicalbros