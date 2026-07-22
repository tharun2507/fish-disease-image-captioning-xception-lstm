

import argparse
import os

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Caption templates. {label} is replaced with the disease/class name derived
# from the image's parent folder, e.g. "Bacterial diseases - Aeromoniasis".
CAPTION_TEMPLATES = [
    "fish image showing {label} condition",
    "aquaculture fish affected by {label}",
    "fish with visible symptoms of {label}",
    "fish disease detection sample showing {label}",
    "fish health analysis image indicating {label}",
]


def clean_label(folder_name: str) -> str:
    """Turn a class folder name into readable caption text."""
    return folder_name.replace("-", " ").replace("_", " ").strip().lower()


def generate_captions(data_dir: str, output_file: str) -> int:
    """Walk data_dir and write templated captions for every image found.

    Returns the number of images captioned.
    """
    count = 0
    with open(output_file, "w") as out:
        for root, _, files in os.walk(data_dir):
            image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
            if not image_files:
                continue

            label = clean_label(os.path.basename(root))

            for file in sorted(image_files):
                for i, template in enumerate(CAPTION_TEMPLATES):
                    caption = template.format(label=label)
                    out.write(f"{file}#{i} {caption}\n")
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate caption annotations for a fish disease image dataset.")
    parser.add_argument("--data_dir", required=True, help="Root directory containing class-labelled image folders.")
    parser.add_argument("--output_file", required=True, help="Path to write the caption annotation file to.")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    n = generate_captions(args.data_dir, args.output_file)
    print(f"Captioned {n} images -> {args.output_file}")


if __name__ == "__main__":
    main()
