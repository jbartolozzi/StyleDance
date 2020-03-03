#!/usr/bin/env bash
python stylegan2/run_projector.py project-real-images \
--network=results/00010-stylegan2-dance2-1gpu-config-f/network-snapshot-000126.pkl \
--dataset=dance_projection \
--data-dir=datasets/processed \
--num-images=6745 \
--result-dir=output/dance_projection