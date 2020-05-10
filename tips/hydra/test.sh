docker container run \
-it \
--mount type=bind,src=$(pwd),dst=/home/src \
hydra-sample