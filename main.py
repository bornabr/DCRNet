import torch
from torch import nn
from dataset import Cost2100DataLoader

import logging

def main():
    
    # Set logging
    
    # Set device
    
    # Load Model
    
    # Load Pretrained Model
    
    # Load Dataset
	train_loader, val_loader, test_loader = Cost2100DataLoader(
	root="./dataset/COST2100",
	batch_size=32,
	num_workers=4,
	pin_memory=False,
	scenario="in")()
 
	# Resume Training
 

	# Train Model
 
	# Evaluate Model
 
	# Save Model

if __name__ == '__main__':
	main()