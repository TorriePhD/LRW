from pathlib import Path
from facetools.detection import transferMotion
import cv2
from cvtransforms import *



# def transferMotion(srcFrame,onlyLips=True,withMouthInternals=False):

def transferMotionVid(vidPath:Path,savePath:Path=None,withMouthInternals=False):
    vidReader = cv2.VideoCapture(str(vidPath))
    imageSize = 96
    vidWriter = cv2.VideoWriter(str(savePath),cv2.VideoWriter_fourcc(*'mp4v'),vidReader.get(cv2.CAP_PROP_FPS),(imageSize,imageSize))
    frames = []
    while True:
        ret,frame = vidReader.read()
        if not ret:
            break
        transfereFrame = transferMotion(frame,withMouthInternals=withMouthInternals)
        if transfereFrame is None:
            frames.append(frame)
            continue
        transfereFrame = cv2.resize(transfereFrame,(imageSize,imageSize))
        frames.append(transfereFrame)
    #fill any None frames with the closest non-None frame
    for i,frame in enumerate(frames):
        if frame is None:
            for j in range(i-1,-1,-1):
                if frames[j] is not None:
                    frames[i] = frames[j]
                    break
            if frames[i] is None:
                for j in range(i+1,len(frames)):
                    if frames[j] is not None:
                        frames[i] = frames[j]
                        break
                
    for frame in frames:
        vidWriter.write(frame)
    vidWriter.release()
    vidReader.release()

if __name__ == "__main__":
    #get the path to directory of videos from args
    #for each video in the directory
    #transfer motion
    #save to new directory
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidDir',type=Path)
    parser.add_argument('--withMouthInternals',action='store_true')

    args = parser.parse_args()
    withMouthInternals = args.withMouthInternals
    vidDir = args.vidDir
    if withMouthInternals:
        saveDir = Path("/home/st392/code/datasets/LRW/lipCropInside_mp4")/vidDir.parent.name/vidDir.name
    else:
        saveDir = Path("/home/st392/code/datasets/LRW/lipCrop_mp4")/vidDir.parent.name/vidDir.name
    saveDir.mkdir(parents=True,exist_ok=True)
    for vidPath in tqdm(list(vidDir.glob('*.mp4'))):
        transferMotionVid(vidPath,saveDir/vidPath.name,withMouthInternals)
