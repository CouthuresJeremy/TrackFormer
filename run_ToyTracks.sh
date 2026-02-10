conda create --name TrackFormer python=3.10 
conda activate TrackFormer
pip install -r requirements.txt  


python main.py fit --config configs/tformerTT.yaml  --data ActsDataModule  --config configs/trainer.yaml
python main.py fit --config configs/tformer.yaml  --data ActsDataModule  --config configs/trainer.yaml
python main.py fit --config configs/tformer.yaml  --data Data/Tml  --config configs/trainer.yaml


python main.py fit --config configs/tformer.yaml


# input env name, if empty, use default acorn
name=acorn
if [ ! -z "$1" ]; then
    name=$1
fi

# this cript is written for a machine with GPU run on CUDA-12. One might need to find out the specific cuda version 
# on their GPU and install the appropriate torch build

conda create --yes --name $name python=3.10 
conda activate $name
conda install --yes pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt  
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 
pip install -e .
python check_acorn.py
