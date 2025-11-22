import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(prog="bsort")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", required=True)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--config", required=True)
    infer_p.add_argument("--image", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.command == "train":
        print("Training with config:", cfg)
        # TODO
    elif args.command == "infer":
        print("Inferencing:", args.image)
        # TODO
    else:
        parser.print_help()
