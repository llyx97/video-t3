git clone https://github.com/NVlabs/VILA.git
cd VILA
git checkout 2527cf75f4fd1f6632aebbf19f27b0c93ac17808
cd ..

vila_path="VILA"

pip install uv
mkdir envs
uv venv envs/.vila
source envs/.vila/bin/activate

uv pip install wheel
uv pip install -e "${vila_path}/."
uv pip install -e "${vila_path}/.[train]"
uv pip install -e "${vila_path}/.[eval]"

uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install packaging
uv pip install ninja
uv pip install flash-attn --no-build-isolation --no-cache-dir
