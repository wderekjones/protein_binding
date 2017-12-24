#!/usr/bin/env bash

source activate protein_binding

python convert_csv_to_h5.py -i data/diverse/parser_output/diverse_subset.csv -o data/diverse/diverse_subset.h5