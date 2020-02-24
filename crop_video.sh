echo $1 $2 $3 $4
docker run --rm -it --runtime=nvidia --volume $PWD:/workspace willprice/nvidia-ffmpeg -y -i $1 -filter:v "crop=$3:$4" -c:a copy $2