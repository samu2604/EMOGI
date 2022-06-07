#!/bin/bash

source $HOME/.bashrc
conda activate hyband-snakemake

hyband generate 6 2 --bracket 0 --last-stage 0 --template-dir $HOME/EMOGI/emogi_random_search/ --output-dir $HOME/EMOGI/emogi_random_search/$1
