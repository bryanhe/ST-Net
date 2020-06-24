#!/bin/bash

cd `python -c "import histonet; print(histonet.config.SPATIAL_RAW_ROOT)"`
for i in */*/*.jpg;
do
    echo ${i}
    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
done
