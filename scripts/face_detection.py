import sys
sys.path.append("../lib")
import utils
import os
import insightface
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def face_detect(image_path, model_retinaface, model_arcface):
    img = cv2.imread(image_path)
    faces, landmarks = model_retinaface.detect(img)
    if len(faces) != 0:
        box = faces[0].astype(np.int).flatten()
        crop_img = img[box[1]:box[3], box[0]:box[2]]
        crop_img = cv2.resize(crop_img, (112, 112))
        emb = model_arcface.get_embedding(crop_img)
        return emb
    return np.zeros((1, 512))

def save_embeddings(frames_path, emb_path, 
                    model_retinaface, model_arcface):
    utils.make_dir(emb_path)
    num_series_of_frames = int(len([name for name in os.listdir(frames_path)]) / 75)
    for i in tqdm(range(num_series_of_frames)):
        file = [name for name in os.listdir(frames_path) if name.startswith(str(i))][0]
        prefix_name = ":".join(file.split(":", 2)[:2])
        embs = np.zeros((75, 1, 512))
        for j in range(1, 76):
            filename = prefix_name + ":{:0>2d}.jpg".format(j)
            embs[i - 1, : ] = face_detect(os.path.join(frames_path, filename), 
                                          model_retinaface, model_arcface)
        np.save(os.path.join(emb_path, "{}.npy".format(prefix_name)), embs)

def init_models(ctx_id, nms):
    model_retinaface = insightface.model_zoo.get_model('retinaface_r50_v1')
    model_retinaface.prepare(ctx_id = args.ctx_id, nms=args.nms)
    model_arcface = insightface.model_zoo.get_model('arcface_r100_v1')
    model_arcface.prepare(ctx_id = args.ctx_id)
    return model_retinaface, model_arcface

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frames_path", action="store",
                        dest="frames_path",
                        help="Path to the folder where frames of video will be saved.")
    parser.add_argument("--emb_path", action="store",
                        dest="emb_path",
                        help="Path to the folder where face embeddings will be saved")
    parser.add_argument("--ctx_id", action="store", type=int, 
                        dest="ctx_id", default=-1, help="If ctx_id is a positive integer then GPUs will be used (default = -1)")
    parser.add_argument("--nms", action="store", type=int,
                        dest="nms", default=0.4, 
                        help="The nms threshold (default=0.4)")
    args = parser.parse_args()

    model_retinaface, model_arcface = init_models(args.ctx_id, args.nms)
    save_embeddings(args.frames_path, args.emb_path, model_retinaface, model_arcface)