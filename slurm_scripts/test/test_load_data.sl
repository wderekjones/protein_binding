#!/bin/bash -l

source activate protein_binding

python -c "from utils.input_pipeline import load_data; X,y = load_data('data/all_kinase/with_pocket/full_kinase_set.h5',label='receptor'); print(X.shape,y.shape); print(y)"