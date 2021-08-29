# Easy Photoz

## Introduction

A command line tool for (PanStarrs)[https://panstarrs.stsci.edu/] Photometric Redshift estimation, using methods similar to [Beck et. al., 2019](https://arxiv.org/pdf/1910.10167.pdf)

## Installation

git clone this repository

```bash
git clone https://github.com/awe2/easy_photoz.git
```
install dependencies

```bash
pip install -r requirements.txt
```

download sfddata-master and place in root of this repo
```bash
wget https://github.com/kbarbary/sfddata/archive/master.tar.gz
tar xzf master.tar.gz
```

Although you do not need it for inference mode, the training data can be grabbed from [here](https://drive.google.com/drive/folders/1tdjogtnGNolk4cV9FWxaMmM5Gunp26Kz?usp=sharing)

## Command Line Tool Usage

example: 
```bash
python easy_photoz_CLI.py 114731807279635598 -f ./../test.npy
```

## General Usage

```python
from easy_photoz.photoz_helper import easy_photoz_objid

objid = [114731807279635598, 2, 114731807279635598] #note misformat objid
#objid = 114731807279635598 #also works fine
posterior, point_estimate, err, returned_objids = easy_photoz_objid(objid)

#posterior, a numpy.ndarray estimate P(z in [z,z+dz]|x); dz = 1.0/360.0
#point estimates, a numpy.ndarray of expectation value from posterior
#err, the error on the point estimate assumping gaussian distribution
#returned objids, if an objid couldn't be matched then its thrown out,
#this returns which objids correspond to each returned value
```

Since the model is just a keras/tensorflow model, you can always load it in your own workflow as...

```python
import tensorflow as tf

mymodel = tf.keras.models.load_model('./MLP.hdf5',custom_objects={'leaky_relu':tf.keras.layers.leaky_relu})

#then do inference as:
posteriors = mymodel(X,training=False)

#see notebooks for details on how to prepare data vector X
```

## Performance
At a glance...

A few different papers had differeing definitions for these values, below I compute with each of these slightly different characteristics.

![MLP Performance](https://github.com/awe2/easy_photoz/tree/main/IMAGES/MLP_performance.PNG)

![MLP PIT](https://github.com/awe2/easy_photoz/tree/main/IMAGES/MLP_PIT.PNG)

```bash
Pasquets Defintions: 
MAD:  0.0155
BIAS:  0.002
ETA:  1.1519 % 5 sigma_mad, percentage
 
Becks Defintions
O:  0.3057 %
MAD:  0.0155
STD:  0.0212
BIAS:  0.0018
 
Tarrio Defintions
STD:  0.0257
BIAS:  0.0018
P0:  Actually they dont well define this metric
 
####################################
#comparison to other works:
 
Tarrio 2020s STD: 0.0298
Tarrio 2020s BIAS: -2.01e-4
 
Beck 2019s O: 1.89%
Beck 2019s MAD: 0.0161
Beck 2019s STD: 0.0322
Beck 2019s BIAS: 5e-4
```

[Pasquet Paper](https://www.aanda.org/articles/aa/full_html/2019/01/aa33617-18/aa33617-18.html) (not a direct comparison, but their paper was highly influential in this work)

[Beck Paper](https://arxiv.org/pdf/1910.10167.pdf) (Most direct comparison. I'm sure our datasets differ slightly, despite my best attempts)

[Tarrio Paper](https://www.aanda.org/articles/aa/full_html/2020/10/aa38415-20/aa38415-20.html)


## Development/About the Author

This code was developed while I was a student at the University of Illinois at Urbana Champaign; I have since begun work as a Data Scientist focussing on Artificial Intelligence at Pacific Northwest National Laboratory. That is to say, I work on this codebase in my spare time and may be slow to respond to requests-- though they are always welcome. I intend this code & dataset to be a launch pad for a few hair-brained ideas I have in Applied AI.

## Kudos

This code was developed while I was a student at the University of Illinois at Urbana Champaign, working as an undergraduate researcher under [Prof. Gautham Narayan](https://gnarayan.github.io/). Similar code (including the model provided here) was implemented as part of the repo [Astro-Ghost](https://github.com/uiucsn/astro_ghost) with the help of [Alexander Gagliano](https://alexandergagliano.github.io/). The model trained here is implemented as part of the [Young Supernovae Experiment's]() Photometric redshift pipeline.

Another kudos should go to [Kyle Barbary](http://kylebarbary.com/) and his [dust maps](https://github.com/kbarbary/sfdmap) repo; which is a dependency here

## Citable

Paper coming soon-- for now please drop a link to this page 
