import os
import shutil
import re

class DatasetSplitter:
    """
    Stratified dataset splitter for bottle-cap classification.
    Splits images and YOLO labels into train/val sets based on color groups.
    """

    def __init__(self, img_dir, lbl_dir, out_img_train, out_img_val, out_lbl_train, out_lbl_val):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.out_img_train = out_img_train
        self.out_img_val = out_img_val
        self.out_lbl_train = out_lbl_train
        self.out_lbl_val = out_lbl_val

        # Mapping from b-code to color group
        self.color_group = {
            "2": "other_b2",
            "3": "other_b3",
            "4": "lightblue_b4",
            "5": "darkblue_b5",
        }

        # Buckets for each b-category
        self.buckets = {
            "other_b2": [],
            "other_b3": [],
            "lightblue_b4": [],
            "darkblue_b5": [],
        }

    def clean_output_dirs(self):
        """Remove old output directories and recreate empty folders."""
        dirs = [
            self.out_img_train, self.out_img_val,
            self.out_lbl_train, self.out_lbl_val
        ]

        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

            os.makedirs(d, exist_ok=True)

        print("Output folders cleaned and recreated.")

    def categorize_files(self):
        """Categorize images into buckets based on filename b-code."""
        for fname in os.listdir(self.img_dir):
            if not fname.endswith(".jpg"):
                continue

            match = re.search(r"_b(\d)_", fname)
            if not match:
                print(f"Skip {fname}: missing b-code pattern.")
                continue

            bcode = match.group(1)
            key = self.color_group.get(bcode)

            if key:
                self.buckets[key].append(fname)
            else:
                print(f"Unknown b-code in: {fname}")

        print("Files categorized into buckets.")

    def split_and_copy(self):
        """
        Perform stratified split: 2 train, 1 val for each bucket.
        Copies corresponding .jpg and .txt files.
        """
        for key, files in self.buckets.items():
            files = sorted(files)

            if len(files) != 3:
                print(f"Warning: {key} expected 3 files, got {len(files)}")

            train_files = files[:2]
            val_files = files[2:]

            # Copy training files
            for f in train_files:
                shutil.copy(os.path.join(self.img_dir, f), self.out_img_train)
                shutil.copy(os.path.join(self.lbl_dir, f.replace(".jpg", ".txt")), self.out_lbl_train)

            # Copy validation files
            for f in val_files:
                shutil.copy(os.path.join(self.img_dir, f), self.out_img_val)
                shutil.copy(os.path.join(self.lbl_dir, f.replace(".jpg", ".txt")), self.out_lbl_val)

        print("Stratified split completed.")

    def run(self):
        """Execute the full pipeline."""
        self.clean_output_dirs()
        self.categorize_files()
        self.split_and_copy()

if __name__ == "__main__":
    splitter = DatasetSplitter(
        img_dir="../data/images_raw",
        lbl_dir="../data/relabels",
        out_img_train="../data/images/train",
        out_img_val="../data/images/val",
        out_lbl_train="../data/labels/train",
        out_lbl_val="../data/labels/val"
    )

    splitter.run()
    print("Process completed.")
