#!/bin/bash

cd $HOME/EMOGI/emogi_random_search/$1/bracket-0/stage-0

best_result=0.0
best_conf=""
current_wd=""

for configuration in $(ls)
do
 cd $configuration
 if $(ls | grep -q 'result'); then
    if (( $(bc <<<"$(cat result) < $best_result") )); then
        best_result=$(cat result);
        best_conf=$(cat config);
        current_wd=$(pwd)
    fi  
 fi  
 cd ..
done

var="text to append";
destdir=$HOME/EMOGI/emogi_random_search/$2
 
echo "The best result is $best_result" >> $destdir
echo "The corresponding configuration is:" >> $destdir
echo $best_conf >> $destdir 
echo "The corresponding configuration directory is: $current_wd" >> $destdir
echo "" >> $destdir

