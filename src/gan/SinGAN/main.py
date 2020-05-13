from config import get_parser


def main():
    args = get_parser().parse_args()
    print(args)


if __name__ == "__main__":
    main()
