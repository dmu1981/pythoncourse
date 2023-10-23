import os
from tqdm import tqdm
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex
import csv


def build(path):
    print("Building from path ", path)

    emb_dim = 512
    index = AnnoyIndex(emb_dim, "angular")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    cnt = 0

    csv_writer = csv.writer(open("index.csv", "wt", newline=""))

    for _, _, files in os.walk(path):
        files = files[:5000]
        bar = tqdm(files)
        for file in bar:
            curpath = os.path.join(path, file)
            bar.set_description(file)

            image = preprocess(Image.open(curpath)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                index.add_item(cnt, image_features[0])

            csv_writer.writerow([cnt, curpath])

            cnt = cnt + 1

    print("Building annoy database")
    index.build(16)
    index.save("annoy.db")
