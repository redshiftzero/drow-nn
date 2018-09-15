# drow-nn

This code generates new [drow elf](http://forgottenrealms.wikia.com/wiki/Drow) names by using the Keras high-level deep learning API to train a LSTM-based language model on a corpus of existing drow names found online.

## Preprocessing data

```
make data
```

Drow names for training were taken from [this list](http://www.dnd.kismetrose.com/DrowNameList.html).

## Training

```
make train
```

This can take a while, and if you want to use early stopping note that you should wait for all weights to be generated prior to sampling from the model.

## Sampling

```
make sample
```

This will generate drow names, e.g.:

```
salaghar daereghel
uldor mulyl
stra yauntyrr
na t'erddrinnshar
zanle abaeir
charsintra yauthlo
yana milyek
im tanor'thal
ldyrrith faertala
beloil yauttir
ondril dlpragh
lin t'xorlarrin
brorn menarn
ch'net melarn
rin menafin
afein ahaurvhel'raugaust
aunirra daeneghel
jyrdyn meerimyder
irra hun'ett
iel dhaulssin
hardsira dhliriy
na mhalazza
barris melarn
ra doyrdlyn
daer yauthlo
uordrin fllifar
uornrae mhalazza
oilrn hllistyn
h tuin'tarl
arbreena t'lodra
ayas melarn
alannora mhlzek
```

## Read More

Sutskever, Martens, and Hinton 2011 - [Generating Text with Recurrent Neural Networks](https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
