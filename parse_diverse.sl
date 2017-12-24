#!/usr/bin/env bash

source activate protein_binding

python full_parser.py --p data/diverse/protein_features/expasy/diverse_expasy.csv data/diverse/protein_features/porter/porter.csv data/diverse/protein_features/profeat/diverse_profeat.csv --pm data/diverse/docking_features_diverse_subset_processed.csv --m data/diverse/dragon7_features/diverse_dragon7_features.csv --out_dir data/diverse/parser_output --o diverse_subset