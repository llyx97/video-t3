import json 
import os 
from tqdm import tqdm 
DATA_DIR = "/home/lilei/Open-LLaVA-NeXT/data/llava-next/"
FILES = [ f for f in os.listdir(DATA_DIR) if "refer-shuffle-order-attribute-1-8x" in f and ( 'total100000' in f or 'total200000' in f) ]
FILES = [ f for f  in FILES if "lRatio2." in f ]

IMAGE_DIR = '/home/lilei/Open-LLaVA-NeXT/data'
def convert_to_llama_factory_sharegpt(data_list):
  converted = [] 
  for sample in tqdm(data_list):
    img_path = [] 
    if sample.get("image", None):
        img_path = [os.path.join(IMAGE_DIR, sample["image"])]
    
    converted_sample = {}
    converted_sample["conversations"] = sample["conversations"]
    converted_sample["images"] = img_path # convert to list 
    converted.append(converted_sample)
  return converted 


for file in FILES:
    print("loading: ", file)
    origin_data = json.load(open(os.path.join(DATA_DIR, file), 'r'))
    converted_data = convert_to_llama_factory_sharegpt(origin_data)
    with open(os.path.join("converted_data", file), 'w') as fw:
      json.dump(converted_data, fw, indent=4)

