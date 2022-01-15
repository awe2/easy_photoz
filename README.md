# Easy Photoz

## Introduction

A command line tool for [PanStarrs](https://panstarrs.stsci.edu/) Photometric Redshift estimation, using methods similar to [Beck et. al., 2019](https://arxiv.org/pdf/1910.10167.pdf)

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

A few different papers had differeing definitions for these values, below I compute each of these slightly different characteristics.

![MLP Performance](/IMAGES/MLP_performance.PNG)

![MLP PIT](/IMAGES/MLP_PIT.PNG)

For an introduction to Photo-Z metrics and this PIT visual metric I show above, [S.J. Schmidt et. al., 2021](https://arxiv.org/pdf/2001.03621.pdf) is a good guide

Before metrics, here is a sample of 4 different posteriors from my network:

![posteriors](/IMAGES/posteriors.PNG)

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

[J. Pasquet, et. al., 2019 Paper](https://www.aanda.org/articles/aa/full_html/2019/01/aa33617-18/aa33617-18.html) (not a direct comparison, but their paper was highly influential in this work)

[R. Beck, et. al., 2019 Paper](https://arxiv.org/pdf/1910.10167.pdf) (Most direct comparison. I'm sure our datasets differ slightly, despite my best attempts)

[P. Tarrio, et. al., 2020 Paper](https://www.aanda.org/articles/aa/full_html/2020/10/aa38415-20/aa38415-20.html)


## Development/About the Author

This code was developed while I was a student at the University of Illinois at Urbana Champaign; I have since begun work as a Data Scientist focussing on Artificial Intelligence at Pacific Northwest National Laboratory. That is to say, I work on this codebase in my spare time and may be slow to respond to requests-- though they are always welcome.

## Kudos

This code was developed while I was a student at the University of Illinois at Urbana Champaign, working as an undergraduate researcher under [Prof. Gautham Narayan](https://gnarayan.github.io/). Similar code (including the model provided here) was implemented as part of the repo [Astro-Ghost](https://github.com/uiucsn/astro_ghost) with the help of [Alexander Gagliano](https://alexandergagliano.github.io/). The model trained here is implemented as part of the [Young Supernovae Experiment's]() Photometric redshift pipeline, among other photo-z methods.

Another kudos should go to [Kyle Barbary](http://kylebarbary.com/) and his [dust maps](https://github.com/kbarbary/sfdmap) repo; which is a dependency here

## Citable

Consider citing GHOST. A indivdual paper for this repository is work in progress.

```bash
@article{GHOST21,
	doi = {10.3847/1538-4357/abd02b},
	url = {https://doi.org/10.3847/1538-4357/abd02b},
	year = 2021,
	month = {feb},
	publisher = {American Astronomical Society},
	volume = {908},
	number = {2},
	pages = {170},
	author = {Alex Gagliano and Gautham Narayan and Andrew Engel and Matias Carrasco Kind and},
	title = {{GHOST}: Using Only Host Galaxy Information to Accurately Associate and Distinguish Supernovae},
	journal = {The Astrophysical Journal},
	abstract = {We present GHOST, a database of 16,175 spectroscopically classified supernovae (SNe) and the properties of their host galaxies. We have constructed GHOST using a novel host galaxy association method that employs deep postage stamps of the field surrounding a transient. Our gradient ascent method achieves fewer misassociations for low-z hosts and higher completeness for high-z hosts than previous methods. Using dimensionality reduction, we identify the host galaxy properties that distinguish SN classes. Our results suggest that the host galaxies of superluminous SNe, Type Ia SNe, and core-collapse SNe can be separated by brightness and derived extendedness measures. Next, we train a random forest model to predict SN class using only host galaxy information and the radial offset of the SN. We can distinguish Type Ia SNe and core-collapse SNe with ∼70\% accuracy without any photometric or spectroscopic data from the event itself. Vera C. Rubin Observatory will usher in a new era of transient population studies, demanding improved photometric tools for rapid identification and classification of transient events. By identifying the host features with high discriminatory power, we will maintain SN sample purities and continue to identify scientifically relevant events as data volumes increase. The GHOST database and our corresponding software for associating transients with host galaxies are both publicly available through the astro_ghost package.}
}
```
