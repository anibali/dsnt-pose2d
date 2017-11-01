#!/usr/bin/env bash

if ! nvidia-docker-compose build
then
    exit 1
fi

servers=(crumb snootles droopy oola)

for server in ${servers[@]}
do
    commie -n -s "$server" -p /home/aiden/Projects/PyTorch/dsnt -c 'nvidia-docker-compose build'
done
