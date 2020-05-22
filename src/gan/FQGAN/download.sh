# get args
DATASET=$1
TARGET_DIR="../../data"
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
    echo  "*" >> "$TARGET_DIR/.gitignore"
    echo  "!.gitignore" >> "$TARGET_DIR/.gitignore"
fi