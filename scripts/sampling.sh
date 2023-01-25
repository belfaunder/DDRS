#!/bin/bash


set -e
#get the root dir (3rd ancestor of the location where this script is stored)
SRC_DIR=`dirname "$BASH_SOURCE"`/../

# Invoke MIP on the rdlp instance files in paralel.
function runBenchmark(){

    #instances=($(cat ./scripts/BAB/instancesHeuristic.lst))
    instances=($(cat ./scripts/BAB/sampling_instances.lst))

    outputdir=./output
    #filename=babSamplingBenchmark.txt

    filename=instance_evaluation_sampling.txt
    #filename=babExact_3_segments.txt
    #filename=Enumeration_3_segments.txt
    #message="Benchmark, 2 h, 20 threads max"
    #header="name	nrCustomers    nrLocations LB   UB	feasible    optimal runtime"
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

    #run the benchmark enumeration
        #python3  ./src/main/discount_strategy/benchmark/enumeration_test.py

    printf "%s\n" "${instances[@]}" |  parallel --timeout 14500 --no-notice  -P 4 -k --eta --colsep ' ' "python3  ./src/main/discount_strategy/test/instance_evaluation_sampling.py {}" >> $outputfile


    #instances=($(cat ./scripts/BAB/instancestemp.lst))
    #printf "%s\n" "${instances[@]}" |  parallel --timeout 7200 --no-notice -P 25 -k --eta --colsep ' ' "python3  ./src/main/discount_strategy/benchmark/ringStar_test.py {}"  >> $outputfile

    #only preserve tagged lines, and remove the tag
    mv $outputfile "${outputfile}_tmp"
    grep ${prefix} "${outputfile}_tmp" | sed -e "s/$prefix//g" > $outputfile
    rm "${outputfile}_tmp"
}

#switch to the root directory. This allows us to invoke this script from any directory. Then run the benchmark.
pushd $SRC_DIR
runBenchmark
popd