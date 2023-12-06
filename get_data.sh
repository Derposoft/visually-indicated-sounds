#!/bin/bash

gdown --fuzzy https://drive.google.com/file/d/11jD2rRA0EXyqkG3yOZ9rgZUoG4gtctc8/view?usp=drive_link
unzip zip.zip
rm -rf data/vig_train
rm -rf data/vig_test
mv zip/* data
rm zip.zip
rm -rf zip
