# Opacity detector


## Introduction
This is a proof of concept on how to dectect opacities on glasses or a plane in the space.
If you just want to see the experiment, please go to the [notebook](Parallax_SIFT.ipynb) . But if you want to run the experiment, follow the next instructions.


## Installation
To install the environment and the required libraries, open a console, go to the project directory and type.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
pip install opencv-contrib-python==3.4.2.16
```
_NOTE_: when the last library is put in the `requirements.txt` file, the notebook will not run. Why?


## Execution
To execute the notebook, open a console, go to the project directory and type.
```bash
virtualenv -p python3 venv
jupyter notebook
```
Then a web browser will open and show the [experiment](http://localhost:8888/notebooks/Parallax_SIFT.ipynb).
