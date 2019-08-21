#!/bin/bash

filenames='fishfilelist.txt'
exec < $filenames

while read line
do
	echo $line
	#echo "${line%.*}.txt"
	python fish.py crop --weights=../../mask_rcnn_fish_0500.h5 --image=$line
done
