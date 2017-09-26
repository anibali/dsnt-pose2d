# Differentiable spatial to numerical transform (DSNT)

This repository contains the DSNT layer and code necessary to train and
evaluate a ResNet-based model on the MPII Human Pose dataset.

## Requirements

### Dataset

Install the `mpii-human-pose` dataset with [DLDS](https://github.com/anibali/dlds).
The default configuration of this project assumes that the dataset will be
installed into `/data`.

Follow the instructions for building the DLDS  Docker image, then run the
following command to install the MPII dataset under `/data`:

```
docker run --rm -it --volume=/data:/data dlds install mpii-human-pose
```

### Showoff (optional)

Showoff is a visualisation server which can be used to display metrics while
training.

To building the Showoff Docker image:

1. Clone the [Showoff repo](https://github.com/anibali/showoff)
2. Change into the `showoff/` directory
3. Build the image with `docker build -t showoff .`

## Running scripts

### Training

1. [Optional] Start the Showoff server.
   ```
   nvidia-docker-compose up showoff -d
   ```
2. Run the training script (pass `--showoff=""` if not using Showoff).
   ```
   nvidia-docker-compose run --rm pytorch bin/train.py --epochs=100
   ```
3. Wait. If using Showoff, you can monitor progress by going to
   [http://localhost:16676](http://localhost:16676).

### Inference

`bin/infer.py` may be used to generate predictions from trained models on the
MPII dataset. The predictions can be written to HDF5 files compatible with
[eval-mpii-pose](https://github.com/anibali/eval-mpii-pose). This is especially
useful for generating Matlab submission files which are compatible with the
official MPII evaluation code.
