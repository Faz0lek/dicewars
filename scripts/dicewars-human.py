#!/usr/bin/env python3
from signal import signal, SIGCHLD
from subprocess import Popen
from time import sleep
from argparse import ArgumentParser


parser = ArgumentParser(prog='Dice_Wars')
parser.add_argument('-n', '--number-of-players', help="Number of players.", type=int, default=2)
parser.add_argument('-b', '--board', help="Seed for generating board", type=int)
parser.add_argument('-s', '--strength', help="Seed for dice assignment", type=int)
parser.add_argument('-o', '--ownership', help="Seed for province assignment", type=int)
parser.add_argument('-p', '--port', help="Server port", type=int, default=5005)
parser.add_argument('-a', '--address', help="Server address", default='127.0.0.1')
parser.add_argument('--ai', help="Specify AI versions as a sequence of ints.", nargs='+')

procs = []


def signal_handler(signum, frame):
    """Handler for SIGCHLD signal that terminates server and clients
    """
    for p in procs:
        try:
            p.kill()
        except ProcessLookupError:
            pass


def main():
    """
    Run the Dice Wars game.

    Example:
        ./dicewars.py -n 4 --ai 4 2 1
        # runs a four-player game with AIs 4, 2, and 1
    """
    args = parser.parse_args()
    ai_versions = [1] * (args.number_of_players - 1)

    signal(SIGCHLD, signal_handler)

    if args.ai:
        if len(args.ai) + 1 > args.number_of_players:
            print("Too many AI versions.")
            exit(1)
        for i in range(0, len(args.ai)):
            ai_versions[i] = args.ai[i]

    try:
        cmd = [
            "./scripts/server.py",
            "-n", str(args.number_of_players),
            "-p", str(args.port),
            "-a", str(args.address),
        ]
        if args.board is not None:
            cmd.extend(['-b', str(args.board)])
        if args.ownership is not None:
            cmd.extend(['-o', str(args.ownership)])
        if args.strength is not None:
            cmd.extend(['-s', str(args.strength)])

        procs.append(Popen(cmd))

        for i in range(1, args.number_of_players + 1):
            if i == 1:
                cmd = [
                    "./scripts/client.py",
                    "-p", str(args.port),
                    "-a", str(args.address),
                ]
            else:
                cmd = [
                    "./scripts/client.py",
                    "-p", str(args.port),
                    "-a", str(args.address),
                    "--ai", str(ai_versions[i - 2]),
                ]

            procs.append(Popen(cmd))
            sleep(0.1)

        for p in procs:
            p.wait()

    except KeyboardInterrupt:
        for p in procs:
            p.kill()


if __name__ == '__main__':
    main()
