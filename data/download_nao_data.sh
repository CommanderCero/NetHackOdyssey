#!/bin/bash

mkdir -p nld_nao
cd nld_nao

# List of zip file suffixes
suffixes=(
  dir-aa dir-ab dir-ac dir-ad dir-ae dir-af dir-ag dir-ah dir-ai dir-aj
  dir-ak dir-al dir-am dir-an dir-ao dir-ap dir-aq dir-ar dir-as dir-at
  dir-au dir-av dir-aw dir-ax dir-ay dir-az dir-ba dir-bb dir-bc dir-bd
  dir-be dir-bf dir-bg dir-bh dir-bi dir-bj dir-bk dir-bl dir-bm dir-bn
  xlogfiles
)

base_url="https://dl.fbaipublicfiles.com/nld/nld-nao"

for suffix in "${suffixes[@]}"; do
  zipfile="nld-nao-${suffix}.zip"
  url="${base_url}/${zipfile}"
  
  echo "Downloading $zipfile..."
  curl -O "$url"
  
  echo "Unzipping $zipfile..."
  unzip -q "$zipfile"
  
  echo "Deleting $zipfile..."
  rm "$zipfile"
done
