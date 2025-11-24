from ultralytics import YOLO

def run_inference(model_path, source, save_dir="runs/bsort"):
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        project=save_dir,
        name="predictions"
    )
    return results
