from unicodedata import numeric
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

import json, os, torch, random, argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    train_feats = train_feats.view(train_feats.shape[0], -1, train_feats.shape[-1])
    val_labels = torch.tensor([[i] * len(feats['val'][category]) for i, category in enumerate(categories)]).flatten().to(device)
    val_feats = torch.cat([feats['val'][category] for category in categories])
    val_feats = val_feats.view(val_feats.shape[0], -1, val_feats.shape[-1])

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

    print(train_feats.shape, train_labels.shape)

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
    print("Aspect:", aspect)
    print(f"Accuracy: {acc:.2f}", num_correct, num_total)
    return acc.cpu()

def encode_single_video(video_path, max_frames_num, model, modalities=['video']):

    disable_torch_init()
    images, num_frames = opencv_extract_frames(video_path, max_frames_num)
    images_tensor = process_images(images, model.get_vision_tower().image_processor, model.config).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        image_features = model.encode_images(images_tensor).to(model.device)
    
    return image_features

def encode_single_video_llm(model, tokenizer, video_path, max_frames_num):
    disable_torch_init()
    images, num_frames = opencv_extract_frames(video_path, max_frames_num)
    images_tensor = process_images(images, model.get_vision_tower().image_processor, model.config).to(model.device, dtype=torch.float16)
    prompt = f"<image>\n"*num_frames
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, [images_tensor])
    with torch.inference_mode():
        outputs = model.llm.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
    return outputs['hidden_states'][-1]

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
    parser.add_argument('--mode', default='feat_extract', choices=['feat_extract', 'probing', 'visualize'])
    parser.add_argument('--feat_type', default='vis', choices=['llm', 'vis'], help='extract visual features from vision encoder or llm')
    parser.add_argument('--probing_model', default='linear', choices=['linear', 'lstm'])
    parser.add_argument('--aspect', default='order_3')
    parser.add_argument('--output_path', default='outputs')
    parser.add_argument('--model_path', default='Efficient-Large-Model/Llama-3-VILA1.5-8B')
    parser.add_argument('--video_root', default='datas/videos')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--max_frames_num", type=int, default=8)
    args = parser.parse_args()

    video_path = aspect2vidpath[args.aspect]
    output_result_path = os.path.join(args.output_path, "probing_results")
    os.makedirs(output_result_path, exist_ok=True)

    categories = os.listdir(os.path.join(args.video_root, video_path['val']))

    if args.mode=='feat_extract':
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

        for data_split in ['train', 'val']:
            for category in categories:
                vfiles = [vf for vf in os.listdir(os.path.join(args.video_root, video_path[data_split], category)) if vf.endswith('.mp4')]
                output_feat_path = os.path.join(args.output_path, "features", "vila", args.feat_type, args.aspect, data_split, category)
                os.makedirs(output_feat_path, exist_ok=True)
                for vfile in tqdm(vfiles):
                    ind = f"{vfile.replace('.mp4', '')}"
                    vfile_ = os.path.join(args.video_root, video_path[data_split], category, vfile)
                    tensor_file = f"{output_feat_path}/{ind}.pt"
                    if os.path.exists(tensor_file):
                        continue
                    if args.feat_type == 'vis':
                        feats = encode_single_video(vfile_, args.max_frames_num, model)
                    elif args.feat_type=='llm':
                        feats = encode_single_video_llm(model, tokenizer, vfile_, args.max_frames_num)
                    torch.save(feats.float().cpu(), tensor_file)
    else:   # load extracted features, for probing only mode
        feats = {}
        for data_split in ['train', 'val']:
            feats[data_split] = {}
            for category in categories:
                aspect = args.aspect.split('_')[0] if 'direction' in args.aspect else args.aspect
                feat_path = os.path.join(args.output_path, "features", "vila", args.feat_type, aspect, data_split, category)
                print(f"Loading features from {feat_path}")
                feat_files = [f"{feat_path}/{file}" for file in os.listdir(feat_path) if file.endswith('.pt')]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    cur_feats = list(tqdm(executor.map(load_feature, feat_files), total=len(feat_files)))
                feats[data_split][category] = torch.stack(cur_feats)
        
        if args.mode=='probing':
            results = {}
            for seed in [12, 22, 32, 42, 52]:
                torch.manual_seed(seed)
                random.seed(seed)
                results[seed] = train_probing(feats, categories, args.probing_model, args.aspect)
            print("Aspect:", args.aspect)
            print(results)
            print(f"Avg Accuracy: {np.mean(list(results.values())):.2f}")
        elif args.mode=='visualize':
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            for token_id in [-1]:
                all_data = torch.cat([feats['val'][key].squeeze(1)[:, token_id, :] for key in feats['val']], dim=0)
                labels = torch.cat([torch.full((feats['val'][key].shape[0],), i) for i, key in enumerate(feats['val'])], dim=0)

                # 将PyTorch tensor转换为NumPy array以便使用sklearn的TSNE
                all_data_np = all_data.numpy()

                # 使用t-SNE降维到2维
                tsne = TSNE(n_components=2, random_state=42)
                data_2d = tsne.fit_transform(all_data_np)

                # 可视化结果
                plt.figure(figsize=(10, 8))
                for i in range(len(feats['val'])):
                    plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1], label=f"{list(feats['val'].keys())[i]}")

                plt.legend()
                plt.title("t-SNE Visualization of 5 Classes")
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                # plt.show()
                plt.savefig(f"tsne_visualization{token_id}.jpg", format='jpg')
