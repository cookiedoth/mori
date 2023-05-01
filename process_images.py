from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import glob
import json
import math
import os
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', choices=['none', 'conc', 'avg'], default='none',
  help='Text features usage: none, conc, avg')
parser.add_argument('-o', '--out_path', default='features.npz')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

print('Loading meta...')
meta = json.loads(open('meta.json').read())

print('Loading images...')
ART_PATH = 'images/saved/*'
art = []

filename_to_id = {}

for i, filename in enumerate(tqdm(sorted(glob.iglob(ART_PATH)))):
  image = Image.open(filename)
  image.load()
  art.append(image)
  filename_to_id[os.path.split(filename)[1]] = i

print('Downloading models...')
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

def image_process(artwork, batch_size=50):
  with torch.no_grad():
    processed_features = []
    for i in tqdm(range(math.ceil(len(artwork) / batch_size))): # len(artwork) // batch_size
      slice_start = i*batch_size
      slice_end = i*batch_size + batch_size if (i*batch_size + batch_size < len(art)) else len(art)
      process_slice = artwork[slice_start : slice_end]
      inputs = processor(images=process_slice, return_tensors="pt").to(device)
      image_features = model.get_image_features(**inputs)
      processed_features.append(image_features)
    output = torch.cat(processed_features, dim=0)
    return output

def text_process(batch_size=50):
  with torch.no_grad():
    processed_features = {}
    embedding_dim = None
    for i in tqdm(range(math.ceil(len(meta) / batch_size))):
      slice_start = i*batch_size
      slice_end = i*batch_size + batch_size if (i*batch_size + batch_size < len(art)) else len(art)
      process_slice = meta[slice_start : slice_end]
      inputs = tokenizer(list(map(lambda x: x['title'], process_slice)), padding=True, return_tensors="pt").to(device)
      text_features = model.get_text_features(**inputs)
      embedding_dim = text_features.shape[1]
      for j in range(len(process_slice)):
        for filename in process_slice[j]['imgs']:
          processed_features[filename_to_id[filename]] = text_features[j]
    result = torch.zeros(len(art), embedding_dim).to(device)
    for i in range(len(art)):
      if i not in processed_features:
        print(f'Cant find the title for artwork {i}')
        continue
      result[i] = processed_features[i]
    return result
  
image_features = image_process(art, batch_size=10)

if args.mode != 'none':
  text_features = text_process(batch_size=10)

if args.mode == 'none':
  features = image_features
elif args.mode == 'conc':
  features = torch.cat((image_features, text_features), dim=1)
elif args.mode == 'avg':
  features = (image_features + text_features) / 2

np.savez(args.out_path, features=features.cpu())
