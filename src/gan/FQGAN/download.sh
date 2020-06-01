# This file don't have to execute if you use 
# torch_mimicry dataset loading

# get args
DATASET=$1
TARGET_DIR="../data/celeba"
IS_ADDED=false

if [ $DATASET == "celeba" ]; then

    if [[ ! -d "$TARGET_DIR/$DATASET/" ]] ; then

        # Aligned CelebA images
        URL=https://www.dropbox.com/s/j6tp062e14gg5yc/img_align_celeba.zip
        ZIP_FILE=$TARGET_DIR/celeba.zip
        mkdir -p $TARGET_DIR
        wget -N $URL -O $ZIP_FILE
        unzip $ZIP_FILE -d $TARGET_DIR
        rm $ZIP_FILE

        IS_ADDED=true

    else
        echo "data is already downloaded in $TARGET_DIR$DATASET/"
    fi

else
    echo "Available arguments are [celeba], [pretrained-celeba-128x128], [pretrained-celeba-256x256]."
    exit 1
fi

# add .gitignore
if [ $IS_ADDED ]; then
    FILE="$TARGET_DIR/.gitignore"
    if [ ! -f "$FILE" ]; then
        echo  "*" >> "$FILE"
        echo  "!.gitignore" >> "$FILE"    
    fi
fi