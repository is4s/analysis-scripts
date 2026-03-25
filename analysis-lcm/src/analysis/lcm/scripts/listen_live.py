#!/usr/bin/env python3
from lcm import LCM


def handle_time_abs(channel, data):
    print(f"Got message on channel '{channel}'")


def main():
    # Set up the lcm connection
    lcm_connection = LCM('tcpq://172.16.16.79')

    # Subscribe to data channels
    lcm_connection.subscribe('/sensor/ins-d/pva', handle_time_abs)

    while lcm_connection.handle() > 0:
        pass


if __name__ == '__main__':
    main()
