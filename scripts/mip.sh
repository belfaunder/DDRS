#!/bin/bash


set -e
#get the root dir (3rd ancestor of the location where this script is stored)
SRC_DIR=`dirname "$BASH_SOURCE"`/../

# Invoke MIP on the rdlp instance files in paralel.
function runBenchmark(){

    #instances=($(cat ./scripts/BAB/instancesHeuristic.lst))
    instances=($(cat ./scripts/BAB/instances_nr_cust_heuristic_small.lst))

    outputdir=./output
    filename=bab_nrCust_small.txt
    prefix="tag: "

    #test whether output dir exists
    if ! [ -d "$outputdir" ]; then
        mkdir ${outputdir}
    fi
    #create the output file and write the message and header to this file
    outputfile=$outputdir/$filename
    touch $outputfile
    echo "${prefix}${message}" >> $outputfile
    echo "${prefix}${header}" >> $outputfile

    printf "%s\n" "${instances[@]}" |  parallel --timeout 10800  --no-notice  -P 5 -k --eta --colsep ' ' "python3  ./src/main/discount_strategy/test/branchAndBound_test.py {}" >> $outputfile
    #only preserve tagged lines, and remove the tag
    mv $outputfile "${outputfile}_tmp"
    grep ${prefix} "${outputfile}_tmp" | sed -e "s/$prefix//g" > $outputfile
    rm "${outputfile}_tmp"



}

#switch to the root directory. This allows us to invoke this script from any directory. Then run the benchmark.
pushd $SRC_DIR
runBenchmark
popd
