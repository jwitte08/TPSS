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
    parser.add_argument(
            '-mth', '--method',
            type=str,
            default='ACP',
            choices=['ACP', 'MCP', 'MVP'],
            help='The smoothing variant'
            )
    args = parser.parse_args()
    return args


def plot_strong_scaling(str_method, str_section):
    eligible_names = ['2MDoFs', '16MDoFs', '134MDoFs', '1GDoFs']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashed']
    test_to_linestyle = {
            name: style for name, style in zip(eligible_names, linestyles)
            }
    markerstyles = ['.', '^', 'x', 'd', 'x']
    test_to_marker = {
            name: style for name, style in zip(eligible_names, markerstyles)
            }
    marker_size = 8
    line_width = 1.5

    options = parse_args()
    root_dir = options.root_dir
    str_xlabel = r'(\d+)procs'
    str_ylabel = r'apply (max)'
    str_tests = r'\d+[kMG]DoFs'
    pattern_file = r'{}_{}_3D_3deg_{}_{}.time\Z'.format(
            str_section,
            str_method,
            str_xlabel,
            str_tests
            )

    orgfiles = [
            os.path.join(root_dir, basename)
            for basename in find_files(root_dir, pattern_file)
            ]

    def yield_testnames():
        pattern = str_tests
        for file in orgfiles:
            match = re.search(pattern, str(file))
            if match:
                yield match.group(0)

    def convert_metric_prefix(name):
        prefix_to_number = {'k': 1e+3, 'M': 1e+6, 'G': 1e+9}
        match = re.search(r'(\d+)([kMG])', name)
        assert match, "No valid prefix: {}".format(name)
        value = float(match.group(1)) * prefix_to_number[match.group(2)]
        return value
    testnames = set(yield_testnames())
    print('Testnames found: {}'.format(testnames))
    testnames = testnames.intersection(eligible_names)
    testnames = sorted(testnames, key=convert_metric_prefix, reverse=False)
    print('Testnames used: {}'.format(testnames))

    fieldnames = [str_ylabel]

    def yield_orgfiles():
        for testname in testnames:
            def yield_per_test():
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
    plt.loglog()
    plt.grid(True)

    #: Insert data
    xydata = list(yield_xydata())
    for xy, name in zip(xydata, testnames):
        x, y = zip(*xy)

        def yield_perfect():
            first, *rest = y
            n = len(x)
            for i in range(n):
                yield first * (0.5**i)
        y_perf = list(yield_perfect())
        plt.plot(x, y,
                 label=name,
                 color='black',
                 marker=test_to_marker[name],
                 markersize=marker_size,
                 linewidth=line_width
                 )
        # linestyle=test_to_linestyle[name],

        plt.plot(x, y_perf, color='grey', linestyle='dotted', linewidth='0.8')

    #: Legend
    # plt.legend()
    return xydata  # todo


def set_xticks(xydata):
    xticks = set()
    for xy in xydata:
        x, y = zip(*xy)
        xticks = xticks.union(set(x))
    xticks = sorted(xticks)
    xticklabels = [str(value) for value in xticks]
    plt.xticks(ticks=xticks, labels=xticklabels)
    plt.tick_params(axis='x', which='minor', bottom=False)


def main():
    options = parse_args()
    method = options.method
    fig = plt.figure()

    ax11 = plt.subplot(221)
    section = r'vmult'
    plt.title("{}".format(section))
    xydata = plot_strong_scaling(str_method=method, str_section=section)
    plt.ylabel("Wall time [s]")

    plt.subplot(222, sharex=ax11, sharey=ax11)
    section = r'smooth'
    plt.title("{}".format(section))
    xydata = plot_strong_scaling(str_method=method, str_section=section)

    ax21 = plt.subplot(223, sharex=ax11)
    section = r'mg'
    plt.title("{}".format(section))
    xydata = plot_strong_scaling(str_method=method, str_section=section)
    plt.ylabel("Wall time [s]")
    plt.xlabel("Number of cores")

    plt.subplot(224, sharex=ax11, sharey=ax21)
    section = r'solve'
    plt.title("{}".format(section))
    xydata = plot_strong_scaling(str_method=method, str_section=section)
    plt.xlabel("Number of cores")

    set_xticks(xydata)
    plt.suptitle("Strong Scaling ({})".format(method))
    handles, labels = ax11.get_legend_handles_labels()
    n_labels = len(labels)
    fig.legend(handles, labels, loc='lower left', ncol=n_labels, mode="expand")
    fig.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    main()

#        plt.yscale('log')
#        plt.xscale('log')
