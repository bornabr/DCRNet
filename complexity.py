import torch
import logging
import os

from model import DCRNet

from omegaconf import DictConfig, OmegaConf
import hydra

import thop

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

	# Load Model
	model = DCRNet(cfg.db.shape, reduction=cfg.reduction, expansion=cfg.expansion).to(device)

	# Calculate Complexity
	input_ = torch.randn([1, 2, 32, 32]).to(device)
	flops, params = thop.profile(model, inputs=(input_,), verbose=False)
	flops, params = thop.clever_format([flops, params], "%.3f")

	log.info(f"Expansions: {cfg.expansion}, Reductions: {cfg.reduction}")
	log.info("FLOPs: {}".format(flops))
	log.info("Params: {}".format(params))
	
if __name__ == "__main__":
	main()