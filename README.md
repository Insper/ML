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

## Deployment

Pushing to `main` triggers the GitHub Actions workflow, which runs `mkdocs gh-deploy` and publishes the site to GitHub Pages (`gh-pages` branch).
