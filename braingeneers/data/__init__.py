import argparse


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert a single neural data file from the default time-first to channels-first format.'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input neural data path/file local or S3'
    )
    parser.add_argument(
        '--output', type=str, required=False, default=None,
        help='Optional output filename, if none is given then the input filename will be modified with ".channels-first.bin"'
    )

    return vars(parser.parse_args())


def main(**kwargs):
    pass


if __name__ == '__main__':
    main(**arg_parser())
