#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

HDF5_DIR="hdf5-hdf5-1_13_2"
wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_2.tar.gz
tar xvfz hdf5-1_13_2.tar.gz
rm hdf5-1_13_2.tar.gz
cd  hdf5-hdf5-1_13_2
./configure --prefix=/usr/local/hdf5 --enable-fortran;
make -j8;
make install;
cd ..;
rm -r  hdf5-hdf5-1_13_2hdf5-hdf5-1_13_2