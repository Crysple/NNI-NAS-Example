# NNI-NAS-Example

## Overview

This is an example of running NAS on [NNI](https://github.com/microsoft/nni) using NNI's NAS interface. The documents about NNI's NAS interface can be found [here](https://github.com/microsoft/nni/blob/master/docs/en_US/GeneralNasInterfaces.md).

## Step

> 1. INSTALL NNI

- ```python3 -m pip install --upgrade nni```

[More information](https://github.com/microsoft/nni/blob/master/docs/en_US/Installation.md)

> 2. RUN NAS

- `cd NAS/data && . download.sh`
- `tar xzf cifar-10-python.tar.gz && mv cifar-batches cifar10`
- `cd .. && nnictl create --config config.yml`
