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
    def specifier_type(string):
        return [int(i) for i in string.split(',')]
    DESC_='''
Plot univariate (generalized) eigenfunctions for biharmonic local solvers on vertex patches. Reads data from
  <dir>/shapefunctions.dat,
  <dir>/EF_<spc>_<n>.dat,
  <dir>/EV_<spc>.dat,
with <n>=0,...,<N>.
'''

    parser = argparse.ArgumentParser(
            description=DESC_,
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            metavar='FILE_PATH',
            default='.',
            help='The file path <dir> to the directory containing all eigenfunction data.'
            )
    parser.add_argument(
            '-N', '--n_dofs',
            type=int,
            metavar='INT',
            default=3,
            help='The number of degrees of freedom <N>.'
            )
    parser.add_argument(
            '-grd', '--plot_grid',
            type=plot_grid_type,
            metavar='STRING',
            default='2x2',
            help='The 2D grid of subplots passed as string <row>x<col> with integers <row> and <col>.'
            )
    parser.add_argument(
            '-spc', '--specifier',
            type=str,
            metavar='COMMA SEPARATED STRING',
            nargs='+',
            default=['Bip_wrt_sqL,Eigenfunctions of $B_{ip} v_i = \lambda_i L^2 v_i$'],
            help='Comma separated specifiers <fsp>,<title> with file specifier <fsp> and suptitle <title>.'
            )

    args = parser.parse_args()
    return args




def main():
    root_dir = parse_args().root_dir
    N = parse_args().n_dofs
    specifiers = parse_args().specifier
    n_pltrows = parse_args().plot_grid[0]
    n_pltcols = parse_args().plot_grid[1]
    #: [20,11.25]inch = [1920,1080]pixel (16:9 resolution)
    my_figsize = [20, 11.25]

    for sp in specifiers:
        fsp, title_eigen= sp.split(',')
        #: plot univariate shape functions on vertex patch
        fig = plt.figure(figsize=my_figsize)
        for i in range(N):
            plt.subplot(n_pltrows,n_pltcols,i+1)
            fpath = os.path.join(root_dir, "phi_{}.dat".format(i))
            xydata = np.genfromtxt(fpath,dtype=float,comments='#')
            # print (xydata)
            x, y = zip(*xydata)
            plt.plot(x,y)
            plt.title("$\phi_{%d}$" % i)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # see evernote
        fig.suptitle("Shape function basis")
        fig.savefig("shapefunctions.svg",papertype='a4')

        #: plot univariate eigendecomposition, this means eigenfunctions and eigenvalues
        fig = plt.figure(figsize=my_figsize)
        for i in range(N):
            plt.subplot(n_pltrows,n_pltcols,i+1)
            fpath = os.path.join(root_dir, "EF_"+fsp+"_{}.dat".format(i))
            xydata = np.genfromtxt(fpath,dtype=float,comments='#')
            # print (xydata)
            x, y = zip(*xydata)
            plt.plot(x,y)
            plt.title("$v_{%d}$" % i)

        plt.subplot(n_pltrows,n_pltcols,N+1)
        fpath = os.path.join(root_dir, "EV_"+fsp+".dat")
        xydata = np.genfromtxt(fpath,dtype=float,comments='#')
        x, y = zip(*xydata)
        plt.plot(x,y,marker='x',linestyle='none')
        plt.title("$\lambda_0$ to $\lambda_{%d}$" % (N-1))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # see evernote
        fig.suptitle(title_eigen)
        fig.savefig("eigenfunctions_"+fsp+".svg",papertype='a4')

        plt.show()

    return 0


if __name__ == '__main__':
    main()