from pathlib import Path
import os

scriptPath = Path("/home/st392/code/learn-an-effective-lip-reading-model-without-pains/scripts/extractLipRegion.sh")
vidDir = Path("/home/st392/code/datasets/LRW/lipread_mp4")

for label in vidDir.iterdir():
    os.system(f"sbatch {scriptPath} {label/'test'}")
    os.system(f"sbatch {scriptPath} {label/'train'}")
    os.system(f"sbatch {scriptPath} {label/'val'}")
    os.system(f"sbatch {scriptPath} {label/'test'} --withMouthInternals")
    os.system(f"sbatch {scriptPath} {label/'train'} --withMouthInternals")
    os.system(f"sbatch {scriptPath} {label/'val'} --withMouthInternals")
