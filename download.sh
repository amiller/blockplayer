#!/bin/bash

wget https://s3.amazonaws.com/blockplayer/datasets.tar.gz
mkdir data
pushd data
tar -xzf ../datasets.tar.gz
popd
