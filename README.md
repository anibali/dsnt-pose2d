# PyTorch DSNT

This repository contains the official implementation of the differentiable
spatial to numerical (DSNT) layer for PyTorch. Also included is the code
necessary to train and evaluate a
[ResNet](https://arxiv.org/abs/1512.03385) or
[Stacked Hourglass](https://arxiv.org/abs/1603.06937) model
on the [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/).

## Requirements

### Dataset

Edit `docker-compose.yml` to set the desired location for the MPII Human Pose
dataset on your computer.

Next, download and install the MPII Human Pose dataset:

```
$ ./run.sh python
>>> from torchdata import mpii
>>> mpii.install_mpii_dataset('/datasets/mpii')
```

## Running scripts

### Tests

```bash
$ ./run.sh pytest
```

### Training

1. [Optional] Start the Showoff server. Showoff is a visualisation server which can be used to
   display metrics while training.
   ```bash
   $ docker-compose up -d showoff
   ```
2. Run the training script (pass `--showoff=""` if not using Showoff).
   ```bash
   $ ./run.sh src/dsnt/bin/train.py --epochs=100
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

[dsnt.nn](src/dsnt/nn.py) is a self-contained module containing all of the
operations required for DSNT, the loss function, and regularization
terms. If you want to build your own model, simply copy that file into
your project and import it to use the functions contained within.

## Other implementations

If you write an implementation of DSNT, please let me know so that I can add it
to the list.

* Tensorflow: [ashwhall/dsnt](https://github.com/ashwhall/dsnt)

## License and citation

(C) 2017-2018 Aiden Nibali

This project is open source under the terms of the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

If you use any part of this work in a research project, please cite the following paper:

```bibtex
@article{nibali2018numerical,
  title={Numerical Coordinate Regression with Convolutional Neural Networks},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1801.07372},
  year={2018}
}
```
