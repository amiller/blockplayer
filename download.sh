#!/bin/bash

#s3cmd sync s3://blockplayer/datasets/ data/archives/

for x in `ls data/archives/ | sed -e s/\.tar\.gz//`
    tar -xzf data/archives/$x.tar.gz
done

#mkdir data
tar -xzf datasets.tar.gz
