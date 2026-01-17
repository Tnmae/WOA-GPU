#!/usr/bin/env bash

mkdir -p build

cd build

cmake -S .. -B .

make

./CPU_WOA
