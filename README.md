Fast Tensor Product Schwarz Smoothers (TPSS)
============================================

TODO ... intro

## Installation

The code is based on the generic finite element library [deal.II](https://github.com/dealii/dealii). Using MPI requires the "forest-of-octrees" library [p4est](https://github.com/cburstedde/p4est) which is responsible to efficiently partition the triangulation with respect to distributed memory.

---
**Warning**

To run the current TPSS software this branch (https://github.com/jwitte08/dealii/tree/rt_matrixfree) of jwitte08's deal.II fork is required.

---

### Installing p4est

First clone **deal.II**, here, e.g. cloned to the directory _dealii_:

```bash
git clone https://github.com/dealii/dealii.git dealii
```
or, alternatively, for ssh access:

```bash
git clone git@github.com:dealii/dealii.git dealii
```

Next, we download **p4est** into a previously created directory _p4est_:

```bash
mkdir p4est
cd p4est
wget http://p4est.github.io/release/p4est-2.0.tar.gz
```

We run the _p4est-setup.sh_ script provided by deal.II to compile the debug (DEBUG) as well as release (FAST) build:

```bash
bash path_to_dealii/doc/external-libs/p4est-setup.sh p4est-2.0.tar.gz `pwd`
```

where `pwd` returns the current working directory _path_to_p4est_. After p4est is built and installed we set an environment variable to _path_to_p4est_:

```bash
export P4EST_DIR=path_to_p4est
```

When deal.II searches for external dependencies it evaluates this variable as hint for p4est.

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

```bash
git clone --recurse-submodules <tpss-repo> <folder-name>
mkdir <folder-name>/build
cd <folder-name>/build
bash ../scripts/tpss-setup.sh
make -j2 all
```

#### Running applications

```bash
<go to build folder>
make -j2 poisson_standard biharmonic_c0ip stokes_raviartthomas
./apps/poisson_standard
./apps/biharmonic_c0ip
./apps/stokes_raviartthomas
```

For the applications in `apps`  a few methodological parameters need to be set at compile time. These parameters are available in `include/ct_parameter.h`.

There exists a helper script `scripts/ct_parameter.py` which can be run with a python3 interpreter. Using

```bash
python3 scripts/ct_parameter.py -h
```

presents an overview of possible options. For example, running

```bash
python3 scripts/ct_parameter.py -DIM 2 -DEG 3 -SMO 'MVP'
```

resets the spatial dimension to 2, finite element degree to 3 and the smoother to a **M**ultiplicative, **V**ertex-**Patch** Schwarz smoother. Note, that the executables need to be rebuilt.

The Poisson problem and some Stokes problems can be executed in parallel via MPI.

```bash
mpirun -np 2 ./apps/poisson_standard
mpirun -np 2 ./apps/stokes_raviartthomas
```

### Running tests

Requires a valid setup of build files (see previous section).

```bash
<go to the build folder>
make -j2 setup_tests
cd tests
ctest --output-on-failure
```
