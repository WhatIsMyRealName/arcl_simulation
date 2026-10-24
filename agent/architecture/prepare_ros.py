"""Prepare causal image/action targets from an incremental ROS session."""
import argparse
from pathlib import Path
from .capture import prepare_session

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speed-linear-limit', type=float, default=0.25)
    parser.add_argument('--speed-angular-limit', type=float, default=1.0)
    args = parser.parse_args()
    print(f'{prepare_session(args.session, args.output, speed_linear_limit=args.speed_linear_limit, speed_angular_limit=args.speed_angular_limit)} cibles ajoutées dans {args.output}')

if __name__ == '__main__':
    main()
