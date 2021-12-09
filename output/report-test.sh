#!/bin/bash

set -ex

python -m xtructure test ./configs/report/rnn.yaml --report-test
python -m xtructure test ./configs/report/transformer.yaml --report-test
python -m xtructure test ./configs/report/cnn-0.yaml --report-test
python -m xtructure test ./configs/report/cnn-std-0.yaml --report-test
python -m xtructure test ./configs/report/cnn-std-1.yaml --report-test
python -m xtructure test ./configs/report/cnn-std-5.yaml --report-test
python -m xtructure test ./configs/report/cnn-std-exp.yaml --report-test
