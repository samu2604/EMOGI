#!/bin/bash

source $HOME/.bashrc
conda activate hyband-snakemake

hyband generate 7 2 --bracket 0 --last-stage 0 --template-dir $HOME/EMOGI/emogi_random_search/ --output-dir $HOME/EMOGI/emogi_random_search/search

conda activate snakemake

nohup snakemake --snakefile $HOME/EMOGI/emogi_random_search/search/Snakefile --latency-wait 60 -j 100 > $HOME/EMOGI/emogi_random_search/search/log.out
