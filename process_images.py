from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch
import glob
import math
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

ART_PATH = 'images/saved/*'
art = []

for filename in sorted(glob.iglob(ART_PATH)):
  image = Image.open(filename)
  art.append(image)

def batched_process(artwork, batch_size=50):
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
  
image_features = batched_process(art, batch_size=100)
np.savez('image_features.npz', features=image_features.cpu())