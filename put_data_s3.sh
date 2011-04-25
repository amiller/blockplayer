#!/bin/bash

# This file is used to split up the datasets into their own archives before
# they are uploaded to S3.

pushd data/sets/ > /dev/null
mkdir -p data/sets/
mkdir -p ../archives
find . -maxdepth 1 -mindepth 1 -type d -not -name data | while read -r set
do
  export set set=`basename $set`
  touch -d "`find $set -exec stat \{} --printf="%y\n" \; | sort -n -r | head -1`" ${set}test
  if [ ${set}test -nt ../archives/$set.tar.gz ]
  then
    ln -sd `pwd`/$set data/sets/$set
    tar -chzvf ../archives/$set.tar.gz data/ > /dev/null
    echo "Compressed dataset \"$set\" to data/archives/$set.tar.gz"
    rm data/sets/$set
  else
    echo "$set.tar.gz up to date!"
  fi
  rm ${set}test
done
rm -rf data
popd > /dev/null
s3cmd -P sync -rr data/archives/ s3://blockplayer/datasets/
echo "Complete!"
