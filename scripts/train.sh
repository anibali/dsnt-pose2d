#!/usr/bin/env bash

nvidia-docker-compose run \
    -e CUDA_VISIBLE_DEVICES=$PALACE_GPU --user=`id -u` --name=`uuidgen` --rm \
    pytorch \
    bin/train.py --showoff=anibali-ltu.duckdns.org:16676 $@
