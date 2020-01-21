# NNI-NAS-Example

## Overview

This is an example of running NAS on [NNI](https://github.com/microsoft/nni) using NNI's NAS interface. The documents about NNI's NAS interface can be found [here](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/AdvancedFeature/GeneralNasInterfaces.md).

## Step

> 1. INSTALL NNI

- ```python3 -m pip install --upgrade nni```


> 2. RUN NAS

- `cd NAS/data && . download.sh`
- `tar xzf cifar-10-python.tar.gz && mv cifar-batches cifar10`
- `cd .. && nnictl create --config config.yml`

## Other branches

There are four branches in this repo. `Master` branch is an example of NAS in classic mode of NNI. `dev-enas` branch is an example of ENAS on NNI(see NNI's doc for different NAS mode. `dev-darts` and `dev-oneshot` are respectively for DARTS and one-shot models but are currently covering only training phase.
