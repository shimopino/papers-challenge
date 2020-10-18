# anime-faces-dataset: http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/

# get dataset name (like "anime-faces")
DATASET=$1
TARGET_DIR="./input"
# if .gitignore doesn't exist, add it
IS_ADDED=false

if [ $DATASET = "anime-face" ]; then

    if [[ ! -d "$TARGET_DIR/animeface-character-dataset/thumb" ]]; then

        # anime-faces
        # http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
        URL=http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
        ZIP_FILE=$TARGET_DIR/animeface-character-dataset.zip
        mkdir -p $TARGET_DIR
        wget -N $URL -O $ZIP_FILE
        unzip $ZIP_FILE -d $TARGET_DIR
        rm -rf $ZIP_FILE

        # add .gitignore
        IS_ADDED=true

    else
        echo "data is already downloaded in $TARGET_DIR/$DATASET"
    fi

else
    echo "Available arguments are [anime-face]"
    exit 1
fi

# add .gitignore
if "$IS_ADDED"; then
    FILE="$TARGET_DIR/.gitignore"
    if [[ ! -f "$FILE" ]]; then
        echo "*" >> "$FILE"
        echo "!.gitignore" >> "$FILE"
    fi
else
    echo ".gitignore is already exist."
fi