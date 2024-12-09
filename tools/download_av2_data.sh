#!/bin/bash

data_path="./data/av2"

mkdir -p "$data_path"

cd "$data_path"

download_and_extract() {
    local url=$1
    local filename=$(basename "$url")

    echo "Downloading $filename..."
    wget "$url"

    if [ ! -f "$filename" ]; then
        echo "Download failed for $filename"
        return 1
    fi

    echo "Extracting $filename..."
    tar -xf "$filename"

    echo "Deleting $filename..."
    rm "$filename"
}

for i in $(seq -w 0 13); do
    url="https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/train-$i.tar"
    download_and_extract "$url"
done

for i in $(seq -w 0 2); do
    url="https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/val-$i.tar"
    download_and_extract "$url"
done

for i in $(seq -w 0 2); do
    url="https://s3.amazonaws.com/argoverse/datasets/av2/tars/sensor/test-$i.tar"
    download_and_extract "$url"
done

echo "Finish downloading and extracting all data."