#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:19:29 2020

@author: jwitte
"""

import argparse
from argparse import RawTextHelpFormatter
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import os



def parse_args():
    def path_type(name):
        path = os.path.abspath(name)
        assert os.path.exists(path), "{} does not exist.".format(path)
        return path
    def plot_grid_type(string):
        return [int(i) for i in string.split('x')]

    parser = argparse.ArgumentParser(
            description='''Plot univariate (generalized) eigenfunctions for biharmonic
            local solvers on vertex patches.''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            default='.',
            help='The directory which contains all eigenfunction data.'
            )
    parser.add_argument(
            '-N', '--n_dofs',
            type=int,
            default=3,
            help='The number of degrees of freedom.'
            )
    parser.add_argument(
            '-grd', '--plot_grid',
            type=plot_grid_type,
            default='2x2',
            help='The 2D grid of subplots. It is modified by a string ROWxCOL with numeric substitutes for ROW and COL.'
            )

    args = parser.parse_args()
    return args




def main():
    root_dir = parse_args().root_dir
    N = parse_args().n_dofs
    n_pltrows = parse_args().plot_grid[0]
    n_pltcols = parse_args().plot_grid[1]

    #: plot univariate shape functions on vertex patch
    fig = plt.figure()
    for i in range(N):
        plt.subplot(n_pltrows,n_pltcols,i+1)
        fpath = os.path.join(root_dir, "phi_{}.dat".format(i))
        xydata = np.genfromtxt(fpath,dtype=float,comments='#')
        # print (xydata)
        x, y = zip(*xydata)
        plt.plot(x,y)
        plt.title("$\phi_{}$".format(i))
    fig.tight_layout()
    fig.suptitle("Shape function basis")
    # fig.savefig("shapefunctions.svg")

    #: plot univariate eigendecomposition, this means eigenfunctions and eigenvalues
    fig = plt.figure()
    for i in range(N):
        plt.subplot(n_pltrows,n_pltcols,i+1)
        fpath = os.path.join(root_dir, "EF_Bip_wrt_sqL_{}.dat".format(i))
        xydata = np.genfromtxt(fpath,dtype=float,comments='#')
        # print (xydata)
        x, y = zip(*xydata)
        plt.plot(x,y)
        plt.title("$v_{}$".format(i))

    plt.subplot(n_pltrows,n_pltcols,N+1)
    fpath = os.path.join(root_dir, "EV_Bip_wrt_sqL.dat")
    xydata = np.genfromtxt(fpath,dtype=float,comments='#')
    x, y = zip(*xydata)
    plt.plot(x,y,marker='x',linestyle='none')
    plt.title("Eigenvalues")
    fig.tight_layout()
    fig.suptitle("Eigenfunctions of $B_{ip} v_i = \lambda_i L^2 v_i$")

    plt.show()

    return 0


if __name__ == '__main__':
    main()