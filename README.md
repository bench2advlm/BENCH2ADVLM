# Bench2ADVLM: A Closed-Loop Benchmark for Vision-Language Models in Autonomous Driving

## Introduction

**BENCH2ADVLM** is the first unified hierarchical closed-loop evaluation framework for real-time, interactive assessment of Vision-Language Models in autonomous driving systems(ADVLMs). This repository provides implementations for evaluating state-of-the-art ADVLMs including [DriveLM](https://github.com/OpenDriveLab/DriveLM), [Dolphins](https://github.com/SaFoLab-WISC/Dolphins), [EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) and [OmniDrive](https://github.com/NVlabs/OmniDrive) in Bench2ADVLM.


## Environment Setup

### Prerequisites
- Linux OS (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 11.8
- Conda package manager

### Installation Steps
To replicate our experimental environment, follow the steps below.
- **STEP 1: Create conda env**
```bash
conda create -n B2ADVLM python=3.8
conda activate B2ADVLM
```
- **STEP 2: Install CUDA toolkit**
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```
- **STEP 3: Install PyTorch**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- **STEP 4: Clone repository**
```bash
git clone https://github.com/xxxxxx.git
```
- **STEP 5:  **
```bash
cd Bench2ADVLM
conda env update -n B2ADVLM -f requirements.yaml
```

- **STEP 6: Set up CARLA 0.9.15**
```bash
# Download carla
mkdir carla
cd carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import 
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
# Add CARLA PythonAPI to PYTHONPATH
export CARLA_ROOT=YOUR_CARLA_PATH
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.8/site-packages/carla.pth
```

## Model Preparation

### DriveLM
Download the pre-trained checkpoints for DriveLM.
```bash
cd ADVLM/DRIVELM
mkdir ckpts 
# download model weights
```
Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Organize the downloaded files in the following structure.
```
/path/to/DRIVELM/ckpts
├── llama_model_weights
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── tokenizer.model
└── checkpoint-7.pth
```


### EM-VLM4AD
Download the model weights for the `T5-Base` version of EM-VLM4AD from [Google Drive](https://drive.google.com/drive/folders/1K61Ou-m5c5UmN2ggT-Huw3rv7PhW5Wft?usp=sharing). Put the folders into the `multi_frame_results` folder. 
```bash
cd ADVLM/VLM4AD
mkdir multi_frame_results
# download model weights
```
Your directory should look like the following:
```
/path/to/EM-VLM4AD
├── multi_frame_results
│   └── T5-Medium
│       └── latest_model.pth        
└── ...
```

### OmniDrive
Download the model checkpoint and pre-trained weights for OmniDrive.
```bash
cd ADVLM/OmniDrive
mkdir ckpts 
# download model weights
```
Download the [OmniDrive checkpoint](https://huggingface.co/exiawsh/OmniDrive/tree/main) to ./ckpts.

Download the pretrained [2D llm weights](https://huggingface.co/exiawsh/pretrain_qformer/tree/main) and [vision encoder + projector weights](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth) to ./ckpts. The vision encoder + projector weights are extracted from ckpts/pretrain_qformer/.

```bash
# OmniDrive checkpoint
huggingface-cli download --resume-download exiawsh/OmniDrive --local-dir /path/to/OmniDrive/ckpts/ --local-dir-use-symlinks False
# 2D llm weights 
huggingface-cli download --resume-download exiawsh/pretrain_qformer --local-dir /path/to/OmniDrive/ckpts/pretrain_qformer --local-dir-use-symlinks False
# eva02_petr_proj.pth
```

Your directory should look like the following:
```
/path/to/OmniDrive/ckpts
├── iter_10548.pth
└── pretrain_qformer
    ├── config.json
    ├── eva02_petr_proj.pth
    ├── generation_config.json
    ├── pytorch_model-00001-of-00002.bin
    ├── pytorch_model-00002-of-00002.bin
    ├── pytorch_model.bin.index.json
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.model
    └── training_args.bin
```

## Directory structure

After installation and preperation, the structure of our code will be as follows:
```
Bench2ADVLM
├── ADVLM/                                     # target ADVLMs(fast system)
│   ├── DOLPHINS/
│   │   ├── configs/
│   │   ├── mllm/
│   │   └── dolphins.py
│   ├── DRIVELM/
│   │   ├── ckpts/
│   │   ├── llama_drivelm/
│   │   └── drivelm.py
│   ├── OmniDrive/
│   │   ├── ckpts/
│   │   ├── mmdetection3d/
│   │   ├── OpenLane-V2/
│   │   ├── projects/
│   │   ├── tools/
│   │   └── omnidrive.py
│   └── VLM4AD/
│       ├── multi_frame_results/
│       ├── modules/
│       └── vlm4ad.py
├── GVLM/                                      # general-purpose VLMs(slow system)
│   ├── llama3/
│   ├── llama_generation_tool.py
│   ├── llama_selection_tool.py
│   ├── LLAVA/
│   ├── llava_generation_tool.py
│   └── llava_selection_tool.py
├── leaderboard/
│   ├── data/                                  # data and logs
│   ├── leaderboard/
│   ├── team_code/                             # agents in CARLA
│   └── scripts/                               # evaluation scripts
├── README.md
├── requirements.yaml
├── scenario_runner/                           # scenario management
└── tools/                                     # evaluation and visualization tools
    ├── ability_benchmark.py
    ├── check_carla.md
    ├── clean_carla.sh
    ├── efficiency_smoothness_benchmark.py
    ├── generate_advlm_video.py
    ├── merge_route_json.py
    ├── split_xml.py
    └── utils.py
```



## Closed-Loop Evaluation

### Multi-GPU Evaluation 

Run closed-loop evaluation on ADVLMs with the following command:

  ```bash
  bash leaderboard/scripts/advlm_scripts/run_evaluation_multi_dolphins.sh
  ```
Before running, configure the evaluation parameters: `TEAM_AGENT`, `TEAM_CONFIG`, `TASK_NUM`, `GPU_RANK_LIST`, `TASK_LIST` in the script.

You can set as following for close-loop evaluation:
```bash
# set the agent
TEAM_AGENT=team_code/dolphins_b2d_agent.py
# set the parsing model [llama/llava]+[generation/selection]
TEAM_CONFIG=llama+generation
```

If you want to test your own model, add your model implementation to `ADVLM/your_model/`, create the agent script in `leaderboard/team_code/your_agent.py`, and modify the evaluation script in `leaderboard/scripts/advlm_scripts/`.

```bash
Bench2ADVLM
├── ADVLM/                                    
│   ├--> link your model folder here
├── GVLM/                                      
├── leaderboard/
│   ├── data/                                 
│   ├── leaderboard/
│   ├── team_code/  
│   │   ├--> add your agent here                          
│   └── scripts/   
│       └── advlm_scripts/  
│           ├--> modify the evaluation script                             
├── ...
```

### Metric
- Basic Performance
```bash
# Merge eval json and get driving score and success rate results
python tools/merge_route_json.py -f your_json_folder/
```
 - Behavioral Quality
```bash
# Get driving efficiency and driving smoothness results
python tools/efficiency_smoothness_benchmark.py -f your_json_folder/merged.json -m your_metric_folder/
```

- Specialized Capabilities
```bash
# Get multi-ability results, including skill scores across lane merging, overtaking, emergency braking, yielding, and traffic sign recognition.
python tools/ability_benchmark.py -r your_json_folder/merged.json
```

### Visualization
Generate annotated videos with decision&control overlays:
```bash
python tools/generate_advlm_video.py -f your_rgb_folder/
```

## Related Resources
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM)
- [Dolphins](https://github.com/SaFoLab-WISC/Dolphins)
- [EM-VLM4AD](https://github.com/akshaygopalkr/EM-VLM4AD) 
- [OmniDrive](https://github.com/NVlabs/OmniDrive)