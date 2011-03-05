tar -czf datasets.tar.gz data/sets
s3cmd -P sync -rr datasets.tar.gz s3://blockplayer/datasets.tar.gz
