## Welcome to use the KGKPD model.

### Description

`train.py` is used to train the KGKPD model for VOC dataset.

`kgkpd.py` is used to configure the parameters for model inference.

`predict.py` is used for model inference.

### Installation

Firstly, we need to clone the repository for the Random Walk with Restart (RWR) implementation, which can be found at GitHub - jinhongjung/pyrwr: Python Implementation for Random Walk with Restart (RWR).

``` 
git clone https://github.com/jinhongjung/pyrwr.git
cd pyrwr
pip install -r requirements.txt
python setup.py install
```

Secondly, Let's start the installation of the KGKPD model.

(if you want to create a new environment for this project, I suggest using python 3.9):
```
conda create --name kgkpd python=3.9
conda activate kgkpd
pip install -r requirements.txt
```

### Train Model
```
python train.py
```

### Predict Image
```
python predict.py
```
