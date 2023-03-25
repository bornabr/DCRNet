import torch
from torch import nn
from dataset import Cost2100DataLoader

import logging
import os

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd

# A logger for this file
log = logging.getLogger(__name__)

def handle_config(cfg):        
	print(OmegaConf.to_yaml(cfg))
	return cfg

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
	
	# Load Config
	cfg = handle_config(cfg)

	# Set device
	if cfg.gpu is None:
		device = torch.device("cpu")
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
		device = torch.device("cuda", cfg.gpu)
		torch.backends.cudnn.benchmark = True
	
	print("Using Device: {}".format(device))
	raise

	# Load Model

	
	# Load Pretrained Model
	
	# Load Dataset
	train_loader, val_loader, test_loader = Cost2100DataLoader(
	root=cfg.db.path,
	batch_size=cfg.batch_size,
	num_workers=cfg.num_workers,
	pin_memory=False,
	scenario=cfg.db.scenario)()
 
	# Resume Training
 

	# Train Model
 
	# Evaluate Model
 
	# Save Model

if __name__ == '__main__':
	main()