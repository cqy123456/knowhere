#!/bin/bash
benchmark_path="$(pwd)/tests"
data_path="$benchmark_path/data"
index_prefix="$benchmark_path/index"
log_prefix="$benchmark_path/log"
declare -A dataset
declare -A metriclist
dataset=(
    ["sift1m"]="http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    ["gist1m"]="http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    ["glove1m"]="http://ann-benchmarks.com/glove-200-angular.hdf5"
    ["movielens10m"]="http://ann-benchmarks.com/movielens10m-jaccard.hdf5"
    ["sift_jaccard"]="sift-4096-hamming.hdf5"
    )
metriclist=(
    ["sift1m"]="L2"
    ["gist1m"]="L2"
    ["glove1m"]="IP"
    ["movielens10m"]="JACCARD"
)
echo ${!dataset[@]}
echo ${dataset[@]}
# entry
while getopts "a:d:t:r:h" opt
do 
    case "$opt" in 
    a) parameterA="$OPTARG" ;;
    d) parameterD="$OPTARG" ;;
    t) parameterT="$OPTARG" ;;
    r) parameterR="$OPTARG" ;;
    h) echo "
    Usage : 
    -a task         : all, build, accuracy, performance
    -d dataset_name : sift1m, gist1m, glove200, movielens10m
    -t index_type   : FLAT, IVFFLAT, IVFPQ, IVFSQ, HNSW, ANNOY, DISKANN
    -r              : remove all, include index and dataset
    "
        exit 0 ;;
    esac
done 

if [ $parameterR ]; then 
    echo "remove benchmark tests folder"
    rm -rf $benchmark_path
    exit 0;
fi

if [ ! $parameterA ] | [ ! $parameterD ] | [ ! $parameterT ]; then 
    echo "paramters is missing, you can use -h to see the parameters. "
    exit 0 ;
fi

# create dir
for dir in $benchmark_path $data_path $index_prefix $log_prefix
do
    if [[ ! -d $dir ]]; then
        mkdir $dir
    fi
done
# download dataset if need
if [ ! -n "${dataset[$parameterD]}" ]
then 
    echo "dataset $parameterD is not found"
    exit 0
else
    hdf5_name="$data_path/${dataset[$parameterD]##*/}"
    if [ ! -f $hdf5_name ] 
    then
        wget ${dataset[$parameterD]}
        cp ${dataset[$parameterD]##*/} $hdf5_name 
        rm ${dataset[$parameterD]##*/}
    else 
        echo "$hdf5_name exists"
    fi
    echo "$index_prefix/$parameterT"
fi 

# run tasks
metric=${metriclist[$parameterD]}

if [ ! "$parameterA" == "all" ]; then
    task_list=($parameterA)
else
    task_list=(build  accuracy performance)
fi

for task in ${task_list[@]}
do
    index_path="$index_prefix/${parameterT}_${parameterD}"
    log_path="$log_prefix/${parameterT}_${parameterD}_${task}"
    echo $index_prefix
    echo $index_path
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;
    if [[  -d $index_path ]]; then
        rm -rf $index_path
    fi
    echo "./../cmake_build/cqy-benchmark/knowhere-benchmark --data_path $hdf5_name --index_path $index_path --index_type $parameterT --metric_type ${metric} --runner_type $task --log_path $log_path"
    ./../cmake_build/cqy-benchmark/knowhere-benchmark --data_path $hdf5_name --index_path $index_path --index_type $parameterT --metric_type ${metric} --runner_type $task --log_path $log_path
done 
    

