SHELL:=/bin/bash
dockerfile_d2 := "dockerfiles/Dockerfile_d2"
tag_name_d2 := "d2:latest"

dockerfile_sg2 := "dockerfiles/Dockerfile_sg2"
tag_name_sg2 := "sg2:latest"

ifndef GPUS
GPUS:=all
endif

ifndef UID
UID:=1000
endif

ifndef GID
GID:=1000
endif


build_d2:
	USER_ID=$(UID) docker build -f ./$(dockerfile_d2) -t $(tag_name_d2) ./

shell_d2:
	docker run -u $(UID):$(GID) -v /Volumes/hqueue:/Volumes/hqueue --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS} -w /data --rm -it -v `pwd`:/data -t $(tag_name_d2) /bin/bash

build_sg2:
	USER_ID=$(UID) docker build -f ./$(dockerfile_sg2) -t $(tag_name_sg2) ./

shell_sg2:
	docker run -u $(UID):$(GID) -v /Volumes/hqueue:/Volumes/hqueue --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPUS} -w /data --rm -it -v `pwd`:/data -t $(tag_name_sg2) /bin/bash

shell_ffmpeg:
	docker run -u $(UID):$(GID) --rm -it --runtime=nvidia --volume $(PWD):/workspace --entrypoint bash willprice/nvidia-ffmpeg

# train_0:
# 	docker run -e NVIDIA_VISIBLE_DEVICES=0 -w /data --rm -it -v `pwd`:/data -t $(tag_name) python3 train_gpu0.py

# train_1:
# 	docker run -e NVIDIA_VISIBLE_DEVICES=1 -w /data --rm -it -v `pwd`:/data -t $(tag_name) python3 train_gpu1.py

# train_2:
# 	docker run -e NVIDIA_VISIBLE_DEVICES=2 -w /data --rm -it -v `pwd`:/data -t $(tag_name) python3 train_gpu2.py

