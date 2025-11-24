import click
import yaml
from ultralytics import YOLO
from .detect import run_inference


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """bsort: Bottle Cap Detection CLI"""
    pass

# ----- TRAIN COMMAND -----
@click.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
def train(config):
    """Train YOLO model using config file."""
    cfg = load_config(config)["train"]

    # Build YOLO command dynamically
    yolo_args = {
        "data": cfg["data"],
        "model": cfg["model"],
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "project": cfg["project"],
        "name": cfg["name"],
        "save": cfg.get("save", True),
    }

    if "freeze" in cfg:
        yolo_args["freeze"] = cfg["freeze"]

    click.echo("Starting training with YOLO...")

    model = YOLO(cfg["model"])
    model.train(**yolo_args)

    click.echo("Training complete!")


# ----- INFER COMMAND -----
@click.command()
@click.option('--config', required=True, help='Path to YAML configuration file')
@click.option('--image', required=True, help='Path to image for inference')
def infer(config, image):
    """Run inference using model & config file."""
    cfg = load_config(config)["infer"]

    model_path = cfg["model"]
    save_dir = cfg["project"] + "/" + cfg["name"]

    click.echo("Running inference...")
    result = run_inference(model_path, image, save_dir)
    click.echo(f"Results saved to: {save_dir}")


cli.add_command(train)
cli.add_command(infer)
