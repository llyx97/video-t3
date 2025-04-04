from operator import truediv
from pickle import FALSE
from unicodedata import numeric
from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from longva.model.language_model.llava_qwen import LlavaQwenConfig
from longva.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import argparse, os, random, json, re
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import concurrent.futures
import numpy as np

def save_json(datas, json_file):
    with open(json_file, 'w') as f:
        datas = json.dump(datas, f, indent=4)

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

class LinearProbe(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
aspect2epoch = {
    "linear": {
        "order": 100,
        "order_3": 200,
        "referring_begin": 200,
        "referring_middle": 200,
        "referring_end": 200,
        "grounding_people": 200,
        "grounding_cats": 200,
        "grounding_bloom": 200,
        "blooming": 200,
        "brightness": 200
    },
    "lstm": {
        "order": 15,
        "order_3": 120,
        "referring_begin": 120,
        "referring_middle": 120,
        "referring_end": 120,
        "grounding_people": 120,
        "grounding_cats": 120,
        "grounding_bloom": 120,
        "blooming": 80,
        "brightness": 80,
    }
}

def train_probing(feats, categories, probing_model, aspect):
    hidden_size = 128  
    output_size = len(categories)    
    num_epochs = aspect2epoch[probing_model][aspect]
    if probing_model=='lstm':
        learning_rate = 0.00005
        batch_size = 64
    elif probing_model=='linear':
        learning_rate = 0.0005
        batch_size = 800

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_labels = torch.tensor([[i] * len(feats['train'][category]) for i, category in enumerate(categories)]).flatten().to(device)
    train_feats = torch.cat([feats['train'][category] for category in categories])
    val_labels = torch.tensor([[i] * len(feats['val'][category]) for i, category in enumerate(categories)]).flatten().to(device)
    val_feats = torch.cat([feats['val'][category] for category in categories])

    if probing_model=='lstm':
        num_tok, input_size = 128, 1024      # compress the number of visual token and dim to this shape
        model = LSTMClassifier(input_size, hidden_size, output_size).to(device)
        train_feats = F.interpolate(train_feats.unsqueeze(0), size=(num_tok, input_size), mode='bilinear').squeeze(0).to(device)
        val_feats = F.interpolate(val_feats.unsqueeze(0), size=(num_tok, input_size), mode='bilinear').squeeze(0).to(device)
    elif probing_model=='linear':
        model = LinearProbe(val_feats.shape[-1], output_size).to(device)
        train_feats = train_feats[:, -1, :].to(device)
        val_feats = val_feats[:, -1, :].to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("Done training")

    model.eval()
    num_correct, num_total = 0, 0
    for i, (inputs, targets) in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            num_correct += (preds==targets).sum()
            num_total += len(preds)
    acc = 100*num_correct/num_total
    print(f"Accuracy: {acc:.2f}", num_correct, num_total)
    return acc.cpu()

def encode_single_video_llm(model, tokenizer, video_path, max_frames_num):
    prompt = f"<image>\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        outputs = model(input_ids, images=[video_tensor], modalities=["video"], return_dict=True)
    return outputs['hidden_states'][0]

def encode_single_video(video_path, max_frames_num, vision_tower, mm_projector, config, modalities=['video']):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = vision_tower.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(vision_tower.device, dtype=torch.float16)
    images = [video_tensor]
    with torch.inference_mode():
        images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        image_features = encode_multimodals(concat_images, video_idx_in_batch, vision_tower, mm_projector, config, split_sizes)

        new_image_features = []
        mm_patch_merge_type = getattr(config, "mm_patch_merge_type", "flat")
        for image_idx, image_feature in enumerate(image_features):
            # rank0_print(f"Initial feature size : {image_feature.shape}")
            if image_idx in video_idx_in_batch:  # video operations
                image_feature = image_feature.flatten(0, 1)
            elif image_feature.shape[0] > 1:
                # base image feature is never used in unires
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                # rank0_print(f"Before pool : {image_feature.shape}")
                height = width = vision_tower.num_patches_per_side
                assert height * width == base_image_feature.shape[0]
                if hasattr(vision_tower, "image_size"):
                    vision_tower_image_size = vision_tower.image_size
                else:
                    raise ValueError("vision_tower_image_size is not found in the vision tower.")
                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                # Assume 2*2 patches
                # After this, [2,2, 24,24, 4096]
                kernel_size = mm_patch_merge_type.split("avgpool")[-1].split("x")[-1]
                kernel_size = 2
                image_feature = image_feature.view(num_patch_height * num_patch_width, height, width, -1) # [4, 24, 24, 4096]
                image_feature = image_feature.permute(0, 3, 1, 2).contiguous() # [4, 4096, 24, 24]
                image_feature = nn.functional.avg_pool2d(image_feature, kernel_size) # [4, 4096, 12, 12]
                image_feature = image_feature.flatten(2, 3) # [4, 4096, 144]
                image_feature = image_feature.permute(0, 2, 1).contiguous() # [4, 144, 4096]
                image_feature = image_feature.flatten(0, 1) # [576, 4096]
                # rank0_print(f"After pool : {image_feature.shape}")
            else:
                # for text only data, there is a placeholder image feature that is actually never used. 
                image_feature = image_feature[0]
                # rank0_print(f"After here : {image_feature.shape}")
            new_image_features.append(image_feature)

        image_features = new_image_features
    return image_features[0]

def get_2dPool(image_feature, vision_tower, config):
    height = width = vision_tower.num_patches_per_side
    num_frames, num_tokens, num_dim = image_feature.shape
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
    if config.mm_spatial_pool_mode == "average":
        image_feature = nn.functional.avg_pool2d(image_feature, config.mm_spatial_pool_stride)
    elif config.mm_spatial_pool_mode == "max":
        image_feature = nn.functional.max_pool2d(image_feature, config.mm_spatial_pool_stride)
    else:
        raise ValueError(f"Unexpected mm_spatial_pool_mode: {config.mm_spatial_pool_mode}")
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    return image_feature

def encode_multimodals(videos_or_images, video_idx_in_batch, vision_tower, mm_projector, config, split_sizes=None):
    videos_or_images_features = vision_tower(videos_or_images)
    per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
    all_videos_or_images_features = []

    for idx, feat in enumerate(per_videos_or_images_features):
        feat = mm_projector(feat)
        # Post pooling
        if idx in video_idx_in_batch:
            feat = get_2dPool(feat, vision_tower, config)
        all_videos_or_images_features.append(feat)
    return all_videos_or_images_features

def load_feature(file):
    return torch.load(file)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--mode', default='feat_extract', choices=['feat_extract', 'probing'])
    parser.add_argument('--feat_type', default='vis', choices=['llm', 'vis'], help='extract visual features from vision encoder or llm')
    parser.add_argument('--probing_model', default='lstm', choices=['linear', 'lstm'])
    parser.add_argument('--aspect', default='order_3')
    parser.add_argument('--output_path', default='outputs')
    parser.add_argument('--model_path', default='lmms-lab/LongVA-7B-DPO')
    parser.add_argument('--video_root', default='datas/videos')
    parser.add_argument('--max_frames_num', default=8, type=int)
    args = parser.parse_args()

    video_path = aspect2vidpath[args.aspect]
    output_result_path = os.path.join(args.output_path, "probing_results")
    os.makedirs(output_result_path, exist_ok=True)

    categories = os.listdir(os.path.join(args.video_root, video_path['val']))

    if args.mode=='feat_extract':
        tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, "llava_qwen", device_map="cuda:0")
        vision_tower = model.model.vision_tower
        mm_projector = model.model.mm_projector

        for data_split in ['train', 'val']:
            for category in categories:
                vfiles = [vf for vf in os.listdir(os.path.join(args.video_root, video_path[data_split], category)) if vf.endswith('.mp4')]
                output_feat_path = os.path.join(args.output_path, "features", "longva", args.feat_type, args.aspect, data_split, category)
                os.makedirs(output_feat_path, exist_ok=True)
                for vfile in tqdm(vfiles):
                    ind = f"{vfile.replace('.mp4', '')}"
                    vfile_ = os.path.join(args.video_root, video_path[data_split], category, vfile)
                    tensor_file = f"{output_feat_path}/{ind}.pt"
                    if os.path.exists(tensor_file):
                        continue
                    if args.feat_type == 'vis':
                        feats = encode_single_video(vfile_, args.max_frames_num, vision_tower, mm_projector, model.config)
                    elif args.feat_type=='llm':
                        feats = encode_single_video_llm(model, tokenizer, vfile_, args.max_frames_num)
                    torch.save(feats.float().cpu(), tensor_file)
    else:   # load extracted features, for probing only mode
        feats = {}
        for data_split in ['train', 'val']:
            feats[data_split] = {}
            for category in categories:
                aspect = args.aspect.split('_')[0] if 'direction' in args.aspect else args.aspect
                feat_path = os.path.join(args.output_path, "features", "longva", args.feat_type, aspect, data_split, category)
                print(f"Loading features from {feat_path}")
                feat_files = [f"{feat_path}/{file}" for file in os.listdir(feat_path) if file.endswith('.pt')]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    cur_feats = list(tqdm(executor.map(load_feature, feat_files), total=len(feat_files)))
                feats[data_split][category] = torch.stack(cur_feats)
        
        results = {}
        for seed in [12, 22, 32, 42, 52]:
            torch.manual_seed(seed)
            random.seed(seed)
            results[seed] = train_probing(feats, categories, args.probing_model, args.aspect)
        print("Aspect:", args.aspect)
        print(results)
        print(f"Avg Accuracy: {np.mean(list(results.values())):.2f}")
