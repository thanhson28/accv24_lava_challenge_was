# accv24_lava_challege_was
This repo for reproduction of WAS team on LAVA challenge

# Install environment
conda create -n was_lava python==3.10
conda activate was_lava
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install pytz
pip install six
pip install flash-attn
pip install optimum
pip install transformers -U
pip install auto-gptq

# reproduction:
- Please just extract dataset from organizer in current folder, the dataset folder file name "LAVA_Challenge_Data":
- Run:
```
python main_lava.py --data_dir ./LAVA_Challenge_Data --output_dir ./output --model_name 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'

python main_lava.py --data_dir ./LAVA_Challenge_Data --output_dir ./output --model_name 'Qwen/Qwen2-VL-7B-Instruct'

python merge.py --output-dir ./output
```
- Final results will be saved as 'Public_assemble.csv' and 'Private_assemble.csv'