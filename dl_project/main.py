from remote_pdb import RemotePdb
from omegaconf import OmegaConf
import socket

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from dl_project.scripts.train_direction_model import DirectionModelTrainer


def get_public_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('10.254.254.254', 1)) #ip address doesn't matter
    public_ip = s.getsockname()[0]
    s.close()
    return public_ip

def connect_remote_pdb():
    ip = get_public_ip()
    RemotePdb(ip, 4444).set_trace()

#Firewall was preventing remote debugging
if __name__ == '__main__':
    connect_remote_pdb()
    trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_trainging.yaml')['training'])
    trainer.train()