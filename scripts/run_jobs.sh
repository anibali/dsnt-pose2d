#!/usr/bin/env bash

servers=(crumb snootles droopy)

# Sync files to servers
for server in ${servers[@]}
do
    commie -s "$server" -p /home/aiden/Projects/PyTorch/dsnt
done

# Run jobs
palace run $@
