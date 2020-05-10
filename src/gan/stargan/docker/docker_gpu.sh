docker container run \
-it \
-d \
--rm \
--mount type=bind,src=$(pwd)/../,dst=/home/src \
--gpus all \
stargan-gpu bash

docker container run \
-d \
-p 6006:6006 \
--rm \
--mount type=bind,src=$(pwd)/../stargan/logs,dst=/logs \
stargan.tensorboard \
tensorboard --logdir=/logs --bind_all