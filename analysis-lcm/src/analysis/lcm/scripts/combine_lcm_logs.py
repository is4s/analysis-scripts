#!/usr/bin/env python3
import argparse
import glob
import os

from lcm import EventLog


def extract_file_number(file_name):
    return int(os.path.splitext(file_name)[1][1:])


# Takes in LCM logfile and writes a new logfile with a modified channel
def main(basename):
    in_logs = glob.glob(f'{basename}.*')
    in_logs = [log for log in in_logs if os.path.splitext(log)[-1] != '.jlp']
    in_logs.sort(key=extract_file_number)

    out_log = EventLog(basename, 'w', True)

    for in_log in in_logs:
        log = EventLog(in_log, 'r')
        print(f'Adding log: {in_log}')

        msg = log.read_next_event()

        while msg is not None:
            out_log.write_event(msg.timestamp, msg.channel, msg.data)
            msg = log.read_next_event()

        log.close()

    out_log.close()
    print(f'Combined log written to {basename}.')


if __name__ == '__main__':
    examples = """
Example: To combine logs foo.00 and foo.01 from directory bar, run:
    python3 combine_lcm_logs.py bar/foo
    """
    parser = argparse.ArgumentParser(
        description="Combine a set of LCM logs that all share the same base name. Assumes all logs are in the same directory. Output filename will be set to the basename passed via the 'basename' argument.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'basename',
        help="""
        Base name of LCM logs to combine. Output file will be given this name.
        Note: if logs are outside current directory, must include path to that
        directory.""",
        type=str,
    )
    args = parser.parse_args()

    basename = os.path.abspath(args.basename)

    if os.path.exists(basename):
        raise ValueError(
            f'File {basename} already exists. Rather than an existing file, please provide just the basename shared by the logs you want to combine.'
        )

    main(basename)
