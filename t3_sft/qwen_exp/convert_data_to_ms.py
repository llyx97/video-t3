import json
import os
from tqdm import tqdm

DATA_DIR = "/home/lilei/Open-LLaVA-NeXT/data/llava-next/"
FILES = [
    f
    for f in os.listdir(DATA_DIR)
    if "refer-shuffle-order-attribute-1-8x" in f
    and ("total100000" in f or "total200000" in f)
]
FILES = [f for f in FILES if "lRatio2." in f]

IMAGE_DIR = "/home/lilei/Open-LLaVA-NeXT/data"


def convert_to_ms_swift(data_list):
    converted = []
    for sample in tqdm(data_list):
        img_path = []
        video_path = []
        if sample.get("image", None):
            img_path = [os.path.join(IMAGE_DIR, sample["image"])]

        converted_sample = {}
        # converted_sample["conversations"] = sample["conversations"]
        history = []
        for idx in range(len(sample["conversations"])):
            if idx % 2 == 0:
                converted_sample["query"] = (
                    sample["conversations"][idx]["value"]
                    .replace("<image>\n", "<image>")
                    .replace("\n<image>", "<image>")
                )
                history.append([sample["conversations"][idx]["value"]])
            else:
                converted_sample["response"] = sample["conversations"][idx]["value"]
                history[-1].append(sample["conversations"][idx]["value"])

        if len(sample["conversations"]) > 2:  # multi-turn
            converted_sample["history"] = history

        converted_sample["images"] = img_path  # convert to list
        converted.append(converted_sample)
    return converted


def converted_sharegpt_to_ms(data_list, video_only=False):
    converted = []
    for sample in tqdm(data_list):
        img_path = []
        video_path = []
        if sample.get("videos", None):

            video_path = sample["videos"]
        if video_only and len(video_path) == 0:
            continue
        history = []
        converted_sample = {}
        for idx in range(len(sample["conversations"])):
            if idx % 2 == 0:
                converted_sample["query"] = (
                    sample["conversations"][idx]["value"]
                    .replace("<video>\n", "<video>")
                    .replace("\n<video>", "<video>")
                )
                history.append([sample["conversations"][idx]["value"]])
            else:
                converted_sample["response"] = sample["conversations"][idx]["value"]
                history[-1].append(sample["conversations"][idx]["value"])

        if len(sample["conversations"]) > 2:  # multi-turn
            converted_sample["history"] = history
        # converted_sample["conversations"] = sample["conversations"]
        converted_sample["images"] = img_path  # convert to list
        converted_sample["videos"] = video_path
        converted.append(converted_sample)
    return converted


def convert_original_to_sharegpt():
    for file in FILES:
        print("loading: ", file)
        fname = file.split(".")[0]
        origin_data = json.load(open(os.path.join(DATA_DIR, file), "r"))
        converted_data = convert_to_ms_swift(origin_data)
        with open(os.path.join("converted_data", f"{fname}.ms.jsonl"), "w") as fw:
            for sample in converted_data:
                fw.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    # convert_original_to_sharegpt()
    video_only = True
    for file in [
        "converted_data/T366K+VRatio1.0.noimage.valid.json",
        "converted_data/T366K+VRatio2.0.noimage.valid.json",
    ]:
        # ["converted_data/video200k+T3LRatio2-200k.noimage.valid.json", "converted_data/video100k+T3LRatio2-100k.noimage.valid.json"]:
        print("loading: ", file)
        # origin_data = json.load(open( file, "r"))
        origin_data = [json.loads(line) for line in open(file, "r")]
        fname = file.split("/")[-1].replace("json", "ms.jsonl")
        converted_data = converted_sharegpt_to_ms(origin_data, video_only=video_only)
        if video_only:
            fname = fname.replace("ms.jsonl", "ms.videoOnly.jsonl")
        with open(os.path.join("converted_data", fname), "w") as fw:
            for sample in converted_data:
                fw.write(json.dumps(sample) + "\n")
    # convert_original_to_sharegpt
