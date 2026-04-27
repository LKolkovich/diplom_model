#!/usr/bin/env python3
import argparse
import cv2
import os
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import app
sys.path.append(str(Path(__file__).parent))

from app.utils.cv_logic import normalize_template

def main():
    parser = argparse.ArgumentParser(description="Normalize agent templates to 512x512 BGRA.")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing templates")
    parser.add_argument("--output", "-o", required=True, help="Output directory for normalized templates")
    parser.add_argument("--size", "-s", type=int, default=512, help="Target size (default: 512)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    count = 0

    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not read {path}")
                continue
                
            normalized = normalize_template(img, target_size=args.size)
            
            # Save as PNG to preserve transparency
            output_path = output_dir / f"{path.stem}.png"
            cv2.imwrite(str(output_path), normalized)
            print(f"Normalized {path.name} -> {output_path.name}")
            count += 1

    print(f"Done. Normalized {count} templates.")

if __name__ == "__main__":
    main()
