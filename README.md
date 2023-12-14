# megatts2
Unoffical implement of Megatts2

## Install mfa
1. conda create -n aligner && conda activate aligner
2. conda install -c conda-forge montreal-forced-aligner=2.2.17


## Prepare dataset
1. Prepare wav and txt files to ./data/wav 
2. Run `python3 make_ds.py --stage 0 --num_workers 4 --wavtxt_path data/wavs --text_grid_path data/textgrids --ds_path data/ds` and stage 1
