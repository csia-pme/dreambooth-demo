echo "# Inference report for $SUBJECT_NAME" >> report.md

FILES="./images/$SUBJECT_NAME/*"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  echo "## $f" >> report.md
  echo "![]($f" + ' "Generated image")' >> report.md
done

cml comment create --target=commit report.md