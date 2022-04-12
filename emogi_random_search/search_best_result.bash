#!/bin/bash

cd $HOME/EMOGI/emogi_random_search/$1/bracket-0/stage-0

best_result=0.0
best_conf=""

for configuration in $(ls)
do
 cd $configuration
 if $(ls | grep -q 'result'); then
    if (( $(bc <<<"$(cat result) < $best_result") )); then
        best_result=$(cat result);
        best_conf=$(cat config);
    fi  
 fi  
 cd ..
done

echo "The best result is" $best_result
echo "The corresponding conficuration is"
echo $best_conf
