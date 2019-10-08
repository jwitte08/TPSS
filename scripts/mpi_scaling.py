#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:51:38 2019

@author: jwitte
"""

import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import operator
import re

from find_files_per_regex import find_files
from orgmode_to_csv import org_to_csv
from orgmode_to_csv import csv_to_nparray


def parse_args():
    def path_type(name):
        path = os.path.abspath(name)
        assert os.path.exists(path), "{} does not exist.".format(path)
        return path

    parser = argparse.ArgumentParser(
            description='''Plot strong scaling results''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            default='.',
            help='The directory containing the strong scaling results'
            )
    args = parser.parse_args()
    return args


def main():
    options = parse_args()
    root_dir = options.root_dir

    #: Find Files
    str_method = r'MVP'
    str_section = r'vmult'
    str_xlabel = r'(\d+)procs'
    str_ylabel = r'apply (max)'
    str_tests = r'(\d+[kMG])DoFs'
    pattern_file = r'{}_{}_3D_3deg_{}_{}.time\Z'.format(
            str_section,
            str_method,
            str_xlabel,
            str_tests
            )

    orgfiles = list(find_files(root_dir, pattern_file))
    def yield_testnames():
        pattern = str_tests
        for file in orgfiles:
            match = re.search(pattern, str(file))
            if match:
                yield match.group(0)
    testnames = set(yield_testnames())
    print('Testnames found: {}'.format(testnames))
    testnames = set(['2MDoFs', '16MDoFs', '134MDoFs', '1GDoFs'])
    print('Testnames used: {}'.format(testnames))
    fieldnames = [str_ylabel]

    def yield_orgfiles():
        for testname in testnames:
            def yield_per_test():
#                orgfiles = find_files(root_dir, pattern_file)
                for file in orgfiles:
                    fname = str(file)
                    if (fname.find(testname) != -1):
                        yield file
            filtered_files = set(yield_per_test())
            yield filtered_files
    orgfiles_per_test = list(yield_orgfiles())

    #: Extract x- and y-Data to be plotted
    def yield_xydata():
        for orgfiles in orgfiles_per_test:
            def yield_xy():
                for file in orgfiles:
                    match = re.search(str_xlabel, str(file))
                    n_procs = int(match.group(1))

                    fname_csv = org_to_csv(file, fieldnames)
                    data = csv_to_nparray(fname_csv,
                                          dtype=np.double,
                                          delimiter=';',
                                          skip_header=1
                                          )
                    os.remove(fname_csv)
                    median = np.median(data, axis=0)
                    print("extracted (x, y) = ({}, {})".format(n_procs, median.tolist()))
                    yield n_procs, median.tolist()

            xy = list(yield_xy())
            xy.sort(key=operator.itemgetter(0))
            yield xy

    #: Create (sub)plot
    ax = plt.subplot(111)
    plt.suptitle("Strong Scaling of {} ({})".format(str_method, str_section))
    plt.loglog()
    plt.grid(True)

    #: Insert data
    xydata = list(yield_xydata())
    for xy in xydata:
        x, y = zip(*xy)

        def yield_perfect():
            first, *rest = y
            n = len(x)
            for i in range(n):
                yield first * (0.5**i)
        y_perf = list(yield_perfect())
        plt.plot(x, y)
        plt.plot(x, y_perf, color='grey', linestyle='dotted', linewidth='0.8')

    #: Label axes
    xy, *rest = xydata
    xticks, _ = zip(*xy)
    xticklabels = [str(value) for value in xticks]
    plt.xticks(ticks=xticks, labels=xticklabels)
    ax.tick_params(axis='x', which='minor', bottom=False)

    #: Legend

    plt.show()
    return 0


if __name__ == '__main__':
    main()

#        plt.yscale('log')
#        plt.xscale('log')
