import os
from tqdm import tqdm
import torch
import clip
from PIL import Image
from matplotlib import pyplot as plt
from annoy import AnnoyIndex
import csv

def lookup(text):
    print(f"Looking up \"{text}\"")

    emb_dim = 512
    index = AnnoyIndex(emb_dim, 'angular')
    index.load("annoy.db")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)    

    text = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text)
    indices = index.get_nns_by_vector(text_features[0], 6)
    
    csv_reader = csv.reader(open("index.csv", "rt"))
    dct = {int(idx): path for idx, path in csv_reader}
    fig, axs = plt.subplots(2,3)
    for cnt, idx in enumerate(indices):
        row = cnt//3
        col = cnt%3
        path = dct[idx]
        img = Image.open(path)
        axs[row,col].imshow(img)

    plt.show()


