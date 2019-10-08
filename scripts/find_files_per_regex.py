#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 08:03:22 2019

@author: jwitte
"""

import argparse
from argparse import RawTextHelpFormatter
import os
import re


def parse_args():
    def path_type(name):
        path = os.path.abspath(name)
        assert os.path.exists(path), "{} does not exist.".format(path)
        return path

    parser = argparse.ArgumentParser(
            description='''Find all files satisfying the given regular expression.''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            'file_pattern',
            type=str,
            help="The regex identifying the desired file(s)"
            )
    parser.add_argument(
            '-dir', '--root_dir',
            type=path_type,
            default='.',
            help='The root directory of the recursive search'
            )
    args = parser.parse_args()
    return args


def find_files(root_dir, pattern):
    def generate_files():
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                fname = os.path.basename(file)
                match = re.search(pattern, fname)
                if match:
                    yield file
    return list(generate_files())

def main():
    options = parse_args()
    input_dir = options.root_dir
    pattern = options.file_pattern
    print(find_files(input_dir,pattern))

    return 0


if __name__ == '__main__':
    main()
