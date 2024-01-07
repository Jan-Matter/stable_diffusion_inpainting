
Setup of Project in Cluster (Described with VSCode)

- environment tested on python: 3.10.12 and linux

- Download VSCode (If not yet done)
- Download Extention Remote - SSH by Microsoft
- Go to ssh tab and create a new connection  with <ethz_username>@student-cluster.inf.ethz.ch use your ETH password for it
- Open Terminal in VSCode
- use "ssh-keygen" command to create ssh keys and add them to your github settings
- Go to the folder where you want to clone the project using cd and mkdir
- command: "git clone git@github.com:Jan-Matter/stable_diffusion_inpainting.git" to clone project
- cd into project

call
- python3 -m venv pyenv
- source ./pyenv/bin/activate
- python -m pip install -r requirements.txt
download model:
- mkdir models/ldm/stable-diffusion-v1
- wget -O models/ldm/stable-diffusion-v1/model.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
download LSUN church dataset:
- mkdir data/lsun_church
- python dl_project/download.py --out_dir "data/lsun_church" --category church_outdoor [-l sample limit]

To run script:
srun --pty -A deep_learning -G 1 -t 60 bash ./scripts/edit.sh

interactive shell (use with care stop session after you don't use it anymore to safe gpu hours!):
-t gives time in minutes
srun --pty -A deep_learning -G 1 -t 60 bash





Meeting Notes 5.1.2024

- Fit Dataset
    - Download it during Running
    - Ask TAs for TMP Space
    - Kaggle
    - TMP DIR
    - Conclusion for now transform dataset locally to fit it on disk and document process (Andras)
    - Debug Celeb (Ning)

- Writing Report (Ning, Arvid)
    - Write Sections as far as possible with curren knowledge
    - Read Related work

Jan
- hyperparam tuning on church
- training on church -> do edits check results
- train on celeb -> compare results with church results






