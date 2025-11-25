import os
import shutil
import re
from tqdm import tqdm

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

        print("[INFO] Output folders cleaned and recreated.\n")

    def categorize_files(self):
        """Categorize images into buckets based on filename b-code."""
        print("[INFO] Categorizing files into buckets...")

        for fname in tqdm(os.listdir(self.img_dir), desc="Categorizing", unit="file"):
            if not fname.endswith(".jpg"):
                continue

            match = re.search(r"_b(\d)_", fname)
            if not match:
                print(f"[WARN] Skip {fname}: missing b-code pattern.")
                continue

            bcode = match.group(1)
            key = self.color_group.get(bcode)

            if key:
                self.buckets[key].append(fname)
            else:
                print(f"[WARN] Unknown b-code: {fname}")

        print("[INFO] Files categorized into buckets.\n")

    def split_and_copy(self):
        """
        Perform stratified split: 2 train, 1 val for each bucket.
        Copies corresponding .jpg and .txt files.
        """
        print("[INFO] Performing stratified split and copying files...")

        for key, files in tqdm(self.buckets.items(), desc="Splitting Buckets"):
            files = sorted(files)

            if len(files) != 3:
                print(f"[WARN] {key} expected 3 files, got {len(files)}")

            train_files = files[:2]
            val_files = files[2:]

            # Copy training files (LABEL FIRST)
            for f in train_files:
                label_src = os.path.join(self.lbl_dir, f.replace(".jpg", ".txt"))
                try:
                    shutil.copy(label_src, self.out_lbl_train)
                    shutil.copy(os.path.join(self.img_dir, f), self.out_img_train)
                except FileNotFoundError:
                    print(f"[WARN] Missing label for {f}. Skipping.")
                    continue

            # Copy validation files (LABEL FIRST)
            for f in val_files:
                label_src = os.path.join(self.lbl_dir, f.replace(".jpg", ".txt"))
                try:
                    shutil.copy(label_src, self.out_lbl_val)
                    shutil.copy(os.path.join(self.img_dir, f), self.out_img_val)
                except FileNotFoundError:
                    print(f"[WARN] Missing label for {f}. Skipping.")
                    continue

        print("\n[INFO] Stratified split completed.\n")

    def run(self):
        """Execute the full pipeline."""
        self.clean_output_dirs()
        self.categorize_files()
        self.split_and_copy()

        # Output summary
        print("====================================================")
        print(" Dataset Split Completed Successfully!")
        print("----------------------------------------------------")
        print(f" Training Images : {os.path.abspath(self.out_img_train)}")
        print(f" Training Labels : {os.path.abspath(self.out_lbl_train)}")
        print("----------------------------------------------------")
        print(f" Validation Images : {os.path.abspath(self.out_img_val)}")
        print(f" Validation Labels : {os.path.abspath(self.out_lbl_val)}")
        print("====================================================\n")

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
