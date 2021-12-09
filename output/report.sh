#!/bin/bash

set -ex

python -m xtructure report ./configs/report/rnn.yaml
python -m xtructure report ./configs/report/transformer.yaml
python -m xtructure report ./configs/report/cnn-0.yaml
python -m xtructure report ./configs/report/cnn-std-0.yaml
python -m xtructure report ./configs/report/cnn-std-1.yaml
python -m xtructure report ./configs/report/cnn-std-5.yaml
python -m xtructure report ./configs/report/cnn-std-exp.yaml
