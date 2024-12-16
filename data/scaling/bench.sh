#! /usr/bin/env bash
# Demo bash script showing how a scaling benchmark might be carried out.

# change to this directory
SCRIPT_DIR=$(realpath $(dirname "$0"))
cd "$(dirname "$0")"
echo ${SCRIPT_DIR}

BINARY=$(realpath ../../build/bin/Release/exatepp_abm)
PARAMS=$(realpath params.csv)
OUTPUT_DIR="outputs"
COUNT=$(($(wc -l <"$PARAMS") - 1))

mkdir -p ${OUTPUT_DIR}

for ((i=0;i<COUNT;i++)); do
    mkdir -p ${OUTPUT_DIR}/${i}
    echo "run ${i} / ${COUNT}"
    echo "  ${BINARY} -i ${PARAMS} -n ${i} -o \"${OUTPUT_DIR}/${i}\""
    ${BINARY} -i ${PARAMS} -n ${i} -o "${OUTPUT_DIR}/${i}"
done