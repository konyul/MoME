# Step-by-step installation instructions

Use MoME with Docker

**a. We provide Docker Image.**
```shell
docker pull kyparkk/mome:python3.8_torch1.11.0_cu113
docker run --gpus all --shm-size=512g -it -v {DATA_DIR}:{DATA_DIR} kyparkk/mome:python3.8_torch1.11.0_cu113 /bin/bash
```

**b. Clone Mask2Map.**
```
git clone https://github.com/konyul/MoME.git
```

**c. Install requirements**
```shell
cd /path/to/MoME
pip install -r requirements.txt

```

**c. Download pre-trained weights**
Download the pretrained weight of the image backbone from https://github.com/hanchaa/MEFormer
```shell
MoME
├─ ckpts
│  ├─ fcos3d_vovnet_imgbackbone-remapped.pth
│  └─ nuim_r50.pth
├─ figures
├─ projects
└─ tools

```