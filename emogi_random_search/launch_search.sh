source $HOME/.bashrc
conda activate snakemake

nohup snakemake --snakefile $HOME/EMOGI/emogi_random_search/$1/Snakefile --latency-wait 60 -j 100 > $HOME/EMOGI/emogi_random_search/$1/log.out
