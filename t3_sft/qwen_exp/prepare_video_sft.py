import json
import os
import glob
import random
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

random.seed(1234)
DATA_DIR = "/home/lilei/LLaVA-Video/LLaVA-Video-178K"

# {'id': 'v_nw8MpBwL508', 'conversations': [{'from': 'human', 'value': '<image>\nWhere does the video take place?'}, {'from': 'gpt', 'value': 'The video takes place in a large indoor arena filled with spectators.'}, {'from': 'human', 'value': 'What is the gymnast wearing?'}, {'from': 'gpt', 'value': 'The gymnast is wearing a purple leotard with white sleeves.'}, {'from': 'human', 'value': 'What event is being showcased in the video?'}, {'from': 'gpt', 'value': "The event being showcased is the '2014 P&G CHAMPIONSHIPS'."}, {'from': 'human', 'value': 'What are some of the movements the gymnast performs on the balance beam?'}, {'from': 'gpt', 'value': 'The gymnast performs a split leap, a handstand, a back walkover, a high leap with one leg extended, a back handspring, and various controlled movements including steps and poses.'}, {'from': 'human', 'value': "How does the audience react to the gymnast's performance?"}, {'from': 'gpt', 'value': "The audience watches intently and remains captivated by the gymnast's performance."}], 'data_source': '1_2_m_academic_v0_1', 'video': 'academic_source/activitynet/v_nw8MpBwL508.mp4'}


# walk through the data directory and find all the json files
def prepare_llava_sft(num=200_000):
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    cnt = 0
    data_list = []
    for subdir in os.listdir(DATA_DIR):
        subdir_json = glob.glob(os.path.join(DATA_DIR, subdir, "*.json"))
        for file in subdir_json:
            # load the json
            origin_data = json.load(open(file, "r"))
            data_list.extend(origin_data)
    # mapping the data to the new format
    random.shuffle(data_list)  # 1.33M
    print(len(data_list))
    data_list = data_list[:num]  # 100K
    return data_list 


def downsample_video(video_file, target_fps=2):
    try:
        basename = os.path.basename(video_file) 
        dirname = os.path.dirname(video_file) 
        output_path = os.path.join(dirname, f"{basename.split('.')[0]}_fps{target_fps}.mp4")
        
        # Skip if output already exists
        if os.path.exists(output_path):
            return output_path
        
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / target_fps)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        print(f"Error processing {video_file}: {str(e)}")
        return None


def process_sample(sample):
    try:
        converted_sample = {}
        video_path = []
        
        if sample.get("video", None):
            video_file = os.path.join(DATA_DIR, sample["video"])
            down_sampled_path = downsample_video(video_file)
            if down_sampled_path:
                video_path = [down_sampled_path]
        
        converted_sample["conversations"] = sample["conversations"]
        for conv in converted_sample["conversations"]:
            if conv["from"] == "human":
                conv["value"] = (
                    conv["value"]
                    .replace("<image>\n", "<video>\n")
                    .replace("\n<image>", "\n<video>")
                )
            elif conv["from"] == "gpt":
                conv["value"] = conv["value"].replace("<image>", "").replace("<video>", "")
                
        converted_sample["videos"] = video_path
        return converted_sample
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None


def convert_to_llama_factory_sharegpt(data_list):
    converted = []
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_sample, sample) for sample in data_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            if result:
                converted.append(result)
    
    return converted


def main():
    # Load and prepare data
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    data_list = []
    for subdir in os.listdir(DATA_DIR):
        subdir_json = glob.glob(os.path.join(DATA_DIR, subdir, "*.json"))
        for file in subdir_json:
            origin_data = json.load(open(file, "r"))
            data_list.extend(origin_data)
    
    random.shuffle(data_list)
    print(f"Total samples: {len(data_list)}")
    data_list = data_list[:50_000]
    
    # Create output directory if it doesn't exist
    os.makedirs("converted_data", exist_ok=True)
    
    # Convert data using parallel processing
    converted_data = convert_to_llama_factory_sharegpt(data_list)
    
    # Save results
    output_path = os.path.join("converted_data", "llava_video_50k_fps2.json")
    with open(output_path, "w") as fw:
        json.dump(converted_data, fw, indent=4)
    print(f"Saved converted data to {output_path}")

def generate_differen_ratio_dataset(ratio_list=[0.2, 0.5, 1.0, 2.0]):
    random.seed(1234)
    source_file = "converted_data/video200k+T3LRatio2-200k.noimage.valid.json"
    with open(source_file, "r") as f:
        source_data = json.load(f) 
    print(len(source_data))

    video_samples = [ sample for sample in source_data if len(sample.get("videos", [])) > 0]
    # print(video_samples[0]) # 266601
    print(len(video_samples)) # 266601 
    t3_samples = [ sample for sample in source_data if len(sample.get("videos", [])) == 0 ]
    print(len(t3_samples))  # 199998 
    random.shuffle(video_samples) 

    for ratio in ratio_list:
        num_video_samples = int(len(t3_samples) * ratio)
        print(f"ratio: {ratio}, num_video_samples: {num_video_samples}")
        target_samples = video_samples[:num_video_samples] + t3_samples
        random.shuffle(target_samples)
        output_path = os.path.join("converted_data", f"T366K+VRatio{ratio}.noimage.valid.json")
        with open(output_path, "w") as fw:
            for sample in target_samples:
                fw.write(json.dumps(sample) + "\n")
        print(f"Saved converted data to {output_path}")    

if __name__ == "__main__":
    # main()
    generate_differen_ratio_dataset()