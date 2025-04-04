# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import os
import os.path as osp
import re
from io import BytesIO

import requests
import torch
from PIL import Image

from tqdm import tqdm
import random, json
from llava.mm_utils import opencv_extract_frames

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def save_json(datas, json_file):
    with open(json_file, 'w') as f:
        datas = json.dump(datas, f, indent=4)

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

aspect2question = {
    "blooming": [
        {
            "question": "Question: Does the flower bloom or unbloom (the video of blooming is played in reverse) in the video?\nOptions: (A) Bloom. (B) Unbloom.\nAnswer the option only. ",
            "answer": {
                "bloom": ["(a)", "(A) Bloom", "bloom", "A"],
                "bloom_reverse": ["(b)", "(B) Unbloom", "unbloom", "does not bloom", "B"]
            },
            "options": ["Bloom", "Unbloom"]
        }
    ],
    "brightness": [
        {
            "question": "Question: Does the brightness increase (brighten) or decrease (darken) in the video?\nOptions: (A) Brighten. (B) Darken.\nAnswer the option only. ",
            "answer": {
                "brighten": ["(a)", "(A) Brighten", "brighten", "brightness increases", "A"],
                "darken": ["(b)", "(B) Darken", "darken", "brightness decreases", "B"]
            },
            "options": ["Brighten", "Darken"]
        }
    ],
    # cat / people 
    "order": [
        {
            "question": "Question: What is the correct order that the items appear in the video?\nOptions:(A) cat, person. (B) person, cat.\nOnly respond with the correct option. ",
            "answer": {
                "cats_people": ["(a)", "(A) cat, person", "cat, person", "A"],
                "people_cats": ["(b)", "(B) person, cat", "person, cat", "B"]
            },
            "options": ["cat, person", "person, cat"] 
        }
    ],
    "order_3": [
        {
            "question": "Question: What is the correct order that the items appear in the video?\nOptions:(A) cat, person, flower. (B) cat, flower, person. (C) person, cat, flower. (D) person, flower, cat. (E) flower, person, cat. (F) flower, cat, person. \nOnly respond with the correct option. ",
            "answer": {
                "cats_people_bloom": ["(a)", "(A) cat, person, flower", "cat, person, flower", "A"],
                "cats_bloom_people": ["(b)", "(B) cat, flower, person", "cat, flower, person", "B"],
                "people_cats_bloom": ["(c)", "(C) person, cat, flower", "person, cat, flower", "C"],
                "people_bloom_cats": ["(d)", "(D) person, flower, cat", "person, flower, cat", "D"],
                "bloom_people_cats": ["(e)", "(E) flower, person, cat", "flower, person, cat", "E"],
                "bloom_cats_people": ["(f)", "(F) flower, cat, person", "flower, cat, person", "F"],
            },
            "options": ["cat, person, flower", "cat, flower, person", "person, cat, flower", "person, flower, cat", "flower, person, cat", "flower, cat, person"]
        },
    ],
    "referring_begin": [
        {
            "question": "Question: Which item is shown in the begin of the video?\nOptions:(A) a person. (B) a cat. (C) a flower.\nAnswer the option only. ",
            "answer": {
                "people": ["(a)", "(A) a person", "a person", "A"],
                "cats": ["(b)", "(B) a cat", "a cat", "B"],
                "bloom": ["(c)", "(C) a flower", "a flower", "C"],
            },
            "options": ["a person", "a cat", "a flower"]
        }
    ],
    "referring_middle": [
        {
            "question": "Question: Which item is shown in the middle of the video?\nOptions:(A) a person. (B) a cat. (C) a flower.\nAnswer the option only. ",
            "answer": {
                "people": ["(a)", "(A) a person", "a person", "A"],
                "cats": ["(b)", "(B) a cat", "a cat", "B"],
                "bloom": ["(c)", "(C) a flower", "a flower", "C"],
            },
            "options": ["a person", "a cat", "a flower"]
        }
    ],
    "referring_end": [
        {
            "question": "Question: Which item is shown in the end of the video?\nOptions:(A) a person. (B) a cat. (C) a flower.\nAnswer the option only. ",
            "answer": {
                "people": ["(a)", "(A) a person", "a person", "A"],
                "cats": ["(b)", "(B) a cat", "a cat", "B"],
                "bloom": ["(c)", "(C) a flower", "a flower", "C"],
            },
            "options": ["a person", "a cat", "a flower"]
        }
    ],
    "grounding_people": [
        {
            "question": "Question: In which part of the video can we see a person?\nOptions:(A) the begin. (B) the middle. (C) the end.\nAnswer the option only. ",
            "answer": {
                "begin": ["(a)", "(A) the begin", "the begin", "A"],
                "middle": ["(b)", "(B) the middle", "the middle", "B"],
                "end": ["(c)", "(C) the end", "the end", "C"],
            },
            "options": ["the begin", "the middle", "the end"]
        }
    ],
    "grounding_cats": [
        {
            "question": "Question: In which part of the video can we see a cat?\nOptions:(A) the begin. (B) the middle. (C) the end.\nAnswer the option only. ",
            "answer": {
                "begin": ["(a)", "(A) the begin", "the begin", "A"],
                "middle": ["(b)", "(B) the middle", "the middle", "B"],
                "end": ["(c)", "(C) the end", "the end", "C"],
            },
            "options": ["the begin", "the middle", "the end"]
        }
    ],
    "grounding_bloom": [
        {
            "question": "Question: In which part of the video can we see a flower?\nOptions:(A) the begin. (B) the middle. (C) the end.\nAnswer the option only. ",
            "answer": {
                "begin": ["(a)", "(A) the begin", "the begin", "A"],
                "middle": ["(b)", "(B) the middle", "the middle", "B"],
                "end": ["(c)", "(C) the end", "the end", "C"],
            },
            "options": ["the begin", "the middle", "the end"]
        }
    ],
}

aspect2vidpath = {
    "blooming": {
        "train": "blooming_black_bg",
        "val": "blooming_white_bg"
    },
    "brightness": {
        "train": "bright_dark_car",
        "val": "bright_dark_cat"
    },
    "order": {
        "train": "order/black_bg",
        "val": "order/white_bg",
    },
    "order_3": {
        "train": "order_3/black_bg",
        "val": "order_3/white_bg",
    },
    "referring_begin": {
        "train": "referring/black_bg/begin",
        "val": "referring/white_bg/begin",
    },
    "referring_middle": {
        "train": "referring/black_bg/middle",
        "val": "referring/white_bg/middle",
    },
    "referring_end": {
        "train": "referring/black_bg/end",
        "val": "referring/white_bg/end",
    },
    "grounding_people": {
        "train": "grounding/black_bg/people",
        "val": "grounding/white_bg/people",
    },
    "grounding_cats": {
        "train": "grounding/black_bg/cats",
        "val": "grounding/white_bg/cats",
    },
    "grounding_bloom": {
        "train": "grounding/black_bg/bloom",
        "val": "grounding/white_bg/bloom",
    },
}

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def inference_single_video(args, video_file, question, tokenizer, model, image_processor):
    # Model
    disable_torch_init()

    images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)

    qs = f"<video>\n {question}"
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # print(images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs

def inference_single_text(args, question, tokenizer, model):
    # Model
    disable_torch_init()

    qs = question
    # print("input: ", qs)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs

def match_answer(response, labels):
    # Normalize the response for comparison
    normalized_response = response.strip().lower()
    
    # Normalize each label in the list
    normalized_labels = [label.strip().lower() for label in labels]
    
    # Check if the normalized response is in the normalized labels list
    if normalized_response in normalized_labels:
        return True
    else:
        return False

def extract_characters_regex(s, options):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
        "The correct order that the items appear in the video is: ",
        "The correct order that the items appear in the video is ",
        "The item shown in the beginning of the video is "
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    # if len(s.split()) > 10 and not re.search("[ABCD]", s):
    #     return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        for opt in options:
            matches = re.search(opt, s)
            if matches is not None:
                return matches[0]
        return ""
    return matches[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--text_qa", action="store_true", help="whether to use frame captions as context")
    parser.add_argument('--caption_path', default='datas/frame_captions')

    parser.add_argument('--aspect', default='order')
    parser.add_argument('--output_path', default='outputs/probing_results')
    parser.add_argument('--video_root', default='datas/videos')
    args = parser.parse_args()

    # fix seed
    torch.manual_seed(42)
    random.seed(42)

    args.caption_path = f"{args.caption_path}/{args.aspect}.json"
    if args.text_qa:
        captions = load_json(args.caption_path)
        args.output_path = f"{args.output_path}/llm"
    else:
        args.output_path = f"{args.output_path}/videollm"
    os.makedirs(args.output_path, exist_ok=True)
    output_file = f"{args.output_path}/vila_{args.aspect}.json"

    if os.path.exists(output_file):
        results = load_json(output_file)
    else:
        results = {}
    video_path = aspect2vidpath[args.aspect]

    categories = os.listdir(os.path.join(args.video_root, video_path['val']))

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    num_correct, num_total = 0, 0
    for category in categories:
        vfiles = [vf for vf in os.listdir(os.path.join(args.video_root, video_path['val'], category)) if vf.endswith('.mp4')]
        for vfile in tqdm(vfiles):
            ind = f"{category}_{vfile.replace('.mp4', '')}"
            if not ind in results:
                qa = aspect2question[args.aspect][0]
                if args.text_qa:
                    cid = f"{category}_{vfile.replace('.mp4', '')}"
                    pred = inference_single_text(args, captions[cid]['prompt'], tokenizer, model)
                    result = {"question": captions[cid]['prompt'], "answer": qa['answer'][category], "category": category, "pred": pred}
                else:
                    pred = inference_single_video(args, os.path.join(args.video_root, video_path['val'], category, vfile), qa['question'], tokenizer, model, image_processor)
                    result = {"question": qa['question'], "answer": qa['answer'][category], "category": category, "pred": pred}

                matched_response =  extract_characters_regex(pred, qa['options'])
                if match_answer(pred.replace(".", ""), result['answer']) or match_answer(matched_response, result['answer']):
                    result['score'] = 1 
                else:
                    print("Wrong prediction: Response:", pred, 'regex response:', matched_response,  "Correct labels: ", result['answer'])
                    result['score'] = 0
                results[ind] = result
            else:
                result = results[ind]
            num_correct += result['score']
            num_total += 1
            if not args.text_qa:
                save_json(results, output_file)
    save_json(results, output_file)
    print("Aspect:", args.aspect)
    print(f"Accuracy: {100*num_correct/num_total:.2f}", num_correct, num_total)
