# =============================================================================
# Created on Sep 28, 2019
#
# @author: jwitte
# =============================================================================

# import codecs
import argparse
from argparse import RawTextHelpFormatter
# import sys
# import shutil
import os
import csv
# import re


def parse_args():
    def csconvert(argument):
        return argument.split(',')
    parser = argparse.ArgumentParser(
            description='''Parsing the CSV output created by likwid-perfctr.''',
            formatter_class=RawTextHelpFormatter
            )
    parser.add_argument(
            'input',
            help='input file'
            )
    args = parser.parse_args()
    return args


def org_to_csv(fname, fieldnames):
    def lines_without_whitespace():
        with open(fname, 'r') as orgfile:
            lines = orgfile.readlines()
            for line in lines:
                yield line.replace(" ", "").replace("\n", "")

    with open('{}.tmp'.format(fname), 'w') as csvfile:
        for line in lines_without_whitespace():
            print(line, sep='\n', file=csvfile)
    fieldnames = [
            name.replace(" ", "")
            for name in fieldnames
            ]

    with open('{}.tmp'.format(fname), 'r') as orgfile:
        rdialect = csv.unix_dialect
        rdialect.delimiter = '|'
        reader = csv.DictReader(orgfile, dialect=rdialect)
        with open('{}.csv'.format(fname), 'w') as csvfile:
            wdialect = csv.unix_dialect
            wdialect.delimiter = ';'
            wdialect.quoting = csv.QUOTE_NONE
            wdialect.strict = True
            writer = csv.DictWriter(
                    csvfile,
                    fieldnames,
                    dialect=wdialect,
                    extrasaction='ignore'
                    )
            writer.writeheader()
            for row in reader:
                writer.writerow(row)

    tmpfile = os.path.abspath('{}.tmp'.format(fname))
    os.remove(tmpfile)


def main():
    options = parse_args()
    fname = options.input
    assert fname.endswith(".org"), "No org-file inserted."
    fieldnames = ['sample', 'setup (max)', 'apply (max)']
    org_to_csv(fname, fieldnames)


if __name__ == '__main__':
    main()
