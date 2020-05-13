FILE=$1

if [ $FILE == "tiny-imagenet-200" ]; then

    # TinyImagenet images and attribute labels
    URL=https://www.dropbox.com/s/w77w9xyni94gzlc/tiny-imagenet-200.zip
    ZIP_FILE=./data/tiny-imagenet-200.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm -rf $ZIP_FILE

else
    echo "Available arguments are [tiny-imagenet-200]"
    exit 1
fi