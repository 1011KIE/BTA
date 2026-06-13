#!/usr/bin/env python3
from bta.cli import build_parser, dry_run


def main():
    args = build_parser().parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        from bta.experiment import run_experiment

        run_experiment(args)


if __name__ == "__main__":
    main()
