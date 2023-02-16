#!/bin/bash

DIR=/home/mayanachesa/Documents/Thesis/riksdagen_anforanden/data/audio
cd $DIR
for f in *; do
cd $f
ls | grep ".*aud_" | xargs -d"\n" rm
cd ..
done

