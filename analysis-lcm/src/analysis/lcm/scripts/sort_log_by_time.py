#!/usr/bin/env python3

import argparse

from analysis.lcm.logfiles import sort_log


def main():
    parser = argparse.ArgumentParser(
        description="Sort LCM log using measurement timestamps. Outputs new log to '<old_log_basename>_sorted.<ext>'"
    )
    parser.add_argument('filepath', help='Full path to LCM Log file', type=str)
    args = parser.parse_args()
    sort_log(args.filepath)


if __name__ == '__main__':
    main()
