#!bin/bash

echo "ls ./images/$SUBJECT_NAME"
ls "./images/$SUBJECT_NAME"

FILES="./images/$SUBJECT_NAME/*/"

# we have one directory for each prompt :
for d in $FILES ; do
    rm -f report.md
    echo "# $d" >> report.md
    for f in $d ; do
        # take action on each file. $f store current file name
        echo "![]($f)" >> report.md
    done
    cml comment create --target=commit --publish report.md
done
