# Machine Learning

Machine Learning course at [Insper](https://www.insper.edu.br/).

Published at: <https://insper.github.io/ML>

## Local development

``` bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdocs serve -o
```

## Figures

Plots are **pre-rendered** — they are not generated at deploy time. Each figure lives next to its source script (`docs/2026.2/classes/<topic>/*.py` → `*.svg`). To regenerate after editing a script:

``` bash
pip install matplotlib numpy pandas scikit-learn   # once, in the venv
./generate-figures.sh
```

Then commit the updated `.svg`/`.html` outputs.

## Deployment

Pushing to `main` triggers the GitHub Actions workflow, which runs `mkdocs gh-deploy` and publishes the site to GitHub Pages (`gh-pages` branch).
