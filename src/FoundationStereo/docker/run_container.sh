docker rm -f foundation_stereo
DIR=$(pwd)/../
xhost +local:docker
docker run --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --network=host \
    --name foundation_stereo  \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp  \
    --ipc=host \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE foundation_stereo:latest bash \
    -c "cd $DIR && bash"