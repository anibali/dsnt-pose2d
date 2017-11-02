# PyTorch DSNT

This repository contains the official implementation of the differentiable
spatial to numerical (DSNT) layer for PyTorch. Also included is the code
necessary to train and evaluate a
[ResNet](https://arxiv.org/abs/1512.03385) or
[Stacked Hourglass](https://arxiv.org/abs/1603.06937) model
on the [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/).

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

1. Clone the [Showoff repository](https://github.com/anibali/showoff)
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
3. Wait until the training finishes. If using Showoff, you can monitor progress by going to
   [http://localhost:16676](http://localhost:16676).

### Inference

`bin/infer.py` may be used to generate predictions from trained models on the
MPII dataset. The predictions can be written to HDF5 files compatible with
[eval-mpii-pose](https://github.com/anibali/eval-mpii-pose). This is especially
useful for generating Matlab submission files which are compatible with the
official MPII evaluation code.

## Building your own models

[dsnt/nn.py](dsnt/nn.py) is a self-contained file containing all of the
operations required for DSNT, the loss function, and regularization
terms. If you want to build your own model, simply copy that file into
your project and import it to use the functions contained within.

## License and citation

(C) 2017 Aiden Nibali

This project is open source under the terms of the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html). If you use any part of this
work in a research project, please cite the following paper:

```bibtex
@misc{
  title="Check back here later"
}
```
