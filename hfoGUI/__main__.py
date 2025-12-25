import sys

from .cli import build_parser, run_hilbert_batch, run_ste_batch, run_mni_batch, run_consensus_batch, run_dl_batch
from .main import run

version = "1.0.8"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'hilbert-batch':
        run_hilbert_batch(args)
    elif args.command == 'ste-batch':
        run_ste_batch(args)
    elif args.command == 'mni-batch':
        run_mni_batch(args)
    elif args.command == 'consensus-batch':
        run_consensus_batch(args)
    elif args.command == 'dl-batch':
        run_dl_batch(args)
    else:
        run()


if __name__ == '__main__':
    main(sys.argv[1:])
