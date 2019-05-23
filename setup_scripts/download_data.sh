#!/bin/bash

# Set root dir
cd ..
root_dir=$(pwd)

mkdir -p $root_dir/data/ && cd $root_dir/data/
cat <<EOF >README.md
This wiki data comes from the [salesforce website](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/#)

It was originally published in:
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models. 2016.

EOF

echo "******************************************"
echo "* Downloading the 2 million token wiki dataset\n"
echo "******************************************"


wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
cd wikitext-2/
mv wiki.test.tokens test.txt
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
cd ..
rm wikitext-2-v1.zip

echo "******************************************"
echo "* Downloading the 100 million token wiki dataset\n"
echo "******************************************"
cd $root_dir/data/
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
cd wikitext-103/
mv wiki.test.tokens test.txt
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
cd ..
rm wikitext-103-v1.zip

echo "******************************************"
echo "* Data Download complete please see the data/README.md file for more details"
echo "******************************************"
