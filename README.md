Fast Tensor Product Schwarz Smoothers (TPSS)
============================================

TODO ... intro

## Installation

The code is based on the generic finite element library [deal.II](https://github.com/dealii/dealii). Using MPI requires the "forest-of-octrees" library [p4est](https://github.com/cburstedde/p4est) which is responsible to efficiently partition the triangulation with respect to distributed memory.

### Installing p4est

First clone **deal.II**, here, e.g. cloned to the directory _dealii_:

```bash
git clone https://github.com/dealii/dealii.git dealii
```
or, alternatively, for ssh access:

```bash
git clone git@github.com:dealii/dealii.git dealii
```

Next, we download **p4est** into the previously created directory _p4est_:

```bash
mkdir p4est
cd p4est
wget http://p4est.github.io/release/p4est-2.0.tar.gz
```

We run the _p4est-setup.sh_ script provided by deal.II to compile the debug (DEBUG) as well as release (FAST) build:

```bash
bash path_to_dealii/doc/external-libs/p4est-setup.sh p4est-2.0.tar.gz `pwd`
```

where `pwd` prints the current working directory, that is _path_to_p4est_. Directly after p4est is build we set an environment variable to the _p4est_ directory:

```bash
export P4EST_DIR=path_to_p4est
```

### Installing deal.II

First clone this project **TPSS**, e.g. into the directory _tpss_, and use the _dealii-setup.sh_ script to build deal.II:

```bash
cd path_to_dealii
mkdir build
cd build
cp path_to_tpss/scripts/dealii-setup.sh .
bash dealii-setup.sh
make -j2
```

Installing the build into a separate directory is possible and explained in deal.II's INSTALL README.

### Configuring TPSS

TODO ...