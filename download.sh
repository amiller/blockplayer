#!/bin/bash

s3cmd sync s3://blockplayer/datasets/ data/archives/

for x in `ls data/archives/ | sed -e s/\.tar\.gz//`
    pushd data/sets/$x
    tar -xzf data/archives/$x.tar.gz
    popd
done

#mkdir data
tar -xzf datasets.tar.gz
