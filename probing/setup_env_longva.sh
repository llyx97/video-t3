git clone https://github.com/EvolvingLMMs-Lab/LongVA.git

longva_path="LongVA"

pip install uv
mkdir envs
uv venv envs/.longva
source envs/.longva/bin/activate

uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e "${longva_path}/longva/.[train]"
uv pip install packaging &&  uv pip install ninja
uv pip install flash-attn --no-build-isolation --no-cache-dir
uv pip install -r $longva_path/requirements.txt
uv pip install httpx==0.23.3
