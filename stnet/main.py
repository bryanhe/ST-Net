import stnet
import sys

def main(args=None):
    """Command-line interface entry point."""
    parser = stnet.parser()
    args = parser.parse_args(args)

    try:
        func = eval(args.func)
    except AttributeError:
        parser.print_help()
        sys.exit(0)

    func(args)

if __name__ == '__main__':
    main()
