import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Simultaneous Training")
    parser.add_argument("--data_dir", default="../data/", help="path to dataset")
    parser.add_argument(
        "--dataset", default="PHOTO", help="type of dataset", choices=["PHOTO"]
    )
    parser.add_argument(
        "--gantype",
        default="zerogp",
        help="type of GAN loss",
        choices=["wgangp", "zerogp", "lsgan"],
    )
    parser.add_argument("--model_name", type=str, default="SinGAN", help="model name")
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64",
    )
    parser.add_argument("--val_batch", default=1, type=int)
    parser.add_argument(
        "--img_size_max", default=250, type=int, help="Input image size"
    )
    parser.add_argument("--img_size_min", default=25, type=int, help="Input image size")
    parser.add_argument(
        "--img_to_use",
        default=-999,
        type=int,
        help="Index of the input image to use < 6287",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "--validation",
        dest="validation",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test model on validation set"
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--gpu", default=None, type=str, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument("--port", default="8888", type=str)
    return parser
