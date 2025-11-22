import os
import re

class LabelRelabeler:
    """
    Relabel YOLO annotation files based on color-group mapping extracted from filename.
    """

    def __init__(self, label_dir, out_dir, color_map):
        self.label_dir = label_dir
        self.out_dir = out_dir
        self.color_map = color_map

        os.makedirs(self.out_dir, exist_ok=True)

    def extract_code(self, filename):
        """Extract b-code (e.g., b2, b3, b4, b5) from filename."""
        match = re.search(r"_b(\d)_", filename)
        if not match:
            return None
        return "b" + match.group(1)

    def relabel_file(self, fname):
        """Relabel a single YOLO label file based on its color group."""
        code = self.extract_code(fname)
        if code is None:
            print(f"Skip: missing b-code in {fname}")
            return

        if code not in self.color_map:
            print(f"Skip: no mapping for {code} in {fname}")
            return

        new_class = self.color_map[code]

        in_path = os.path.join(self.label_dir, fname)
        out_path = os.path.join(self.out_dir, fname)

        with open(in_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts) + "\n")

        with open(out_path, "w") as f:
            f.writelines(new_lines)

        print(f"Relabeled {fname} → class={new_class}")

    def run(self):
        """Process all label files in the directory."""
        for fname in os.listdir(self.label_dir):
            if fname.endswith(".txt"):
                self.relabel_file(fname)

        print("Relabeling completed.")

if __name__ == "__main__":
    COLOR_MAP = {
        "b2": 2,   # green → other
        "b3": 2,   # orange → other
        "b4": 0,   # light blue
        "b5": 1,   # dark blue
    }

    relabeler = LabelRelabeler(
        label_dir="../data/labels_raw",
        out_dir="../data/labels",
        color_map=COLOR_MAP
    )

    relabeler.run()
