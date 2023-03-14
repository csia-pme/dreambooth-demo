#!bin/bash

rm -f report.md
echo "# Experiment reproduction report" >> report.md
echo "## Parameters used" >> report.md
echo "\`\`\`yaml" >> report.md
cat params.yaml >> report.md
echo "\`\`\`" >> report.md
echo "## Results" >> report.md
echo "![](./images/grid.jpg)" >> report.md
cml comment create --target=commit --publish report.md

