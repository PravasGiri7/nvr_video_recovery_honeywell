import os
import re
import glob
import shutil
import subprocess


INPUT_DIR = r"PATH\TO\INPUT\DIR"
MERGED_FILE = r"PATH\TO\MERGED\FILE.dat"
OUTPUT_FILE = r"PATH\TO\OUTPUT\FILE.mp4"
VIDEO_CODEC = "libx265" #Libx264 when necessary
AUDIO_CODEC = "aac"
CRF = 23
PRESET = "medium"

DELETE_MERGED = True


def natural_sort_key(path: str):
    name = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', name)]


def merge_dat_files():
    dat_files = glob.glob(os.path.join(INPUT_DIR, "*.dat"))
    dat_files.sort(key=natural_sort_key)

    if not dat_files:
        raise FileNotFoundError(f".dat file not found in: {INPUT_DIR}")

    print("[INFO] Merging files:")
    for f in dat_files:
        print(" -", f)

    os.makedirs(os.path.dirname(MERGED_FILE), exist_ok=True)

    with open(MERGED_FILE, "wb") as outfile:
        for dat_file in dat_files:
            with open(dat_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"[INFO] Merging completed: {MERGED_FILE}")


def encode_with_ffmpeg():
    cmd = [
        r"PATH\TO\FFMPEG\ffmpeg.exe",
        "-y",
        "-i", MERGED_FILE,
        "-c:v", VIDEO_CODEC,
        "-c:a", AUDIO_CODEC,
        "-preset", PRESET,
        "-crf", str(CRF),
        OUTPUT_FILE
    ]

    print("[INFO] ffmpeg execution:")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError("ffmpeg encoding failed")

    print(f"[INFO] Encoding completed: {OUTPUT_FILE}")


def main():
    merge_dat_files()
    encode_with_ffmpeg()

    if DELETE_MERGED and os.path.exists(MERGED_FILE):
        os.remove(MERGED_FILE)
        print(f"[INFO] merged file deleted: {MERGED_FILE}")


if __name__ == "__main__":
    main()