# Install
```
  # LongVA
  bash setup_env_longva.sh
  # VILA
  bash setup_env_vila.sh
```
# Data Preparation
- Download the videos and frame captions from [here](https://huggingface.co/datasets/lyx97/t3_probing_data)
- Unzip the video files and place them into `datas/videos`
- Unzip the frame caption file and place it into `datas/frame_captions`
  
# Probing Video LLM
- VILA
```
  python3 vila_inference.py \
    --conv-mode llama_3 \
    --aspect grounding_cats
```
- LongVA
```
  python3 longva_inference.py \
    --aspect grounding_cats
```
# Probing LLM Decoder
- VILA
```
  python3 vila_inference.py \
    --conv-mode llama_3 \
    --aspect grounding_cats \
    --text_qa
```
- LongVA
```
  python3 longva_inference.py \
    --aspect grounding_cats \
    --text_qa
```
# Probing Visual Features
### Step1: Extract Visual Features
- `--feat_type` determines the type of visual feature to extract, where `vis` means the features from vision encoder and `llm` means the features from the last layer of llm hidden state.
```
  # LongVA
  python3 longva_probing.py    \
    --mode feat_extract \
    --feat_type vis \
    --aspect grounding_cats

  # VILA
  python3 vila_probing.py    \
    --mode feat_extract \
    --feat_type vis \
    --aspect grounding_cats
```
### Step2: Train Classifier Probe
- `--probing_model` determines the type of classifier probe (lstm or linear).
```
  # LongVA
  python3 longva_probing.py    \
    --mode probing \
    --feat_type vis \
    --aspect grounding_cats   \
    --probing_model lstm

  # VILA
  python3 vila_probing.py    \
    --mode feat_extract \
    --feat_type vis \
    --aspect grounding_cats   \
    --probing_model lstm
```
