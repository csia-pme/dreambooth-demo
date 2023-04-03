#!bin/bash

rm -f report.md
echo "# Experiment reproduction report" >> report.md
echo "## Params workflow vs. main" >> report.md
echo >> report.md
dvc params diff main --show-md >> report.md
echo "## Input images" >> report.md
echo "![](./data/source_images_grid.jpg)" >> report.md
echo "## Results" >> report.md
echo "![](./images/grid.jpg)" >> report.md
cml comment create --target=commit --publish report.md