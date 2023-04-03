import torch
from torch import nn
import numpy as np

from dataset import Cost2100DataLoader

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import _LRScheduler

import logging
import os

from datetime import datetime

from model import DCRNet

from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path, get_original_cwd

# A logger for this file
log = logging.getLogger(__name__)

def handle_config(cfg):        
	print(OmegaConf.to_yaml(cfg))
	return cfg

def evaluator(pred, data):
    with torch.no_grad():
        # De-centralize
        data = data - 0.5
        pred = pred - 0.5

        # Calculate the NMSE
        power_gt = data[:, 0, :, :] ** 2 + data[:, 1, :, :] ** 2
        difference = data - pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
        
        return nmse

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + np.cos(np.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
	
	# Load Config
	cfg = handle_config(cfg)

	time = datetime.now().strftime("%Y%m%d-%H%M%S")

	if not cfg.evaluate:
		# Set up Tensorboard
		writer = SummaryWriter(f'{cfg.tensorboard_dir}/v{cfg.version}-{time}')

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
	
	# Load Dataset
	if cfg.db.name == 'small':
		data = np.load(cfg.db.path)
		train_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=cfg.batch_size, shuffle=True)
		val_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=cfg.batch_size, shuffle=True)
	else:
		train_loader, val_loader, test_loader = Cost2100DataLoader(
		root=cfg.db.path,
		batch_size=cfg.batch_size,
		num_workers=cfg.num_workers,
		pin_memory=False,
		scenario=cfg.db.scenario)()
 
	# Resume Training
	# TODO: Implement this

	# Train Model
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
	scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
											 T_max=cfg.epochs * len(train_loader),
											 T_warmup= 30 * len(train_loader),
											 eta_min=5e-5)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
	# 													   T_max=cfg.epochs * len(train_loader),
	# 													   eta_min=5e-5)

	best_nmse = 10
	best_epoch = -1
	
	if cfg.evaluate:
		# Load Model
		state = torch.load(os.path.join(cfg.checkpoints_dir, cfg.test_model_name))
		model.load_state_dict(state['state_dict'])

		# Test Model
		test_loss, nmse = test(test_loader, model, device, criterion)
		log.info('====> Test NMSE: {:.3e}'.format(nmse))
		return

	for epoch in range(cfg.epochs):
		train_loss = train(cfg, writer, epoch, train_loader, model, device, criterion, optimizer, scheduler)
		if epoch % cfg.val_interval == 0 or epoch == cfg.epochs - 1:
			val_loss, nmse = validate(writer, epoch, val_loader, model, device, criterion)

			if nmse <= best_nmse:
				best_nmse = nmse
				best_epoch = epoch
				# Save Model
				if cfg.save:
					state = {
						'epoch': epoch,
						'state_dict': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'scheduler': scheduler.state_dict(),
						'best_nmse': best_nmse,
						'best_epoch': best_epoch
					}
					torch.save(state, os.path.join(cfg.checkpoints_dir, f'v{cfg.version}-{time}-best_model.pth'))	

			# Log Best Epoch, Best NMSE
			log.info('====> Best Epoch: {}\tBest NMSE: {:.3e}'.format(best_epoch, best_nmse))

def train(cfg, writer, epoch, train_loader, model, device, criterion, optimizer, scheduler):
	model.train()
	train_loss = 0
	num_batches = len(train_loader)
	for batch_idx, (data,) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, data)
		loss.backward()
		optimizer.step()
		scheduler.step()
		train_loss += loss.item()
		if batch_idx % cfg.log_interval == 0 or batch_idx == num_batches - 1:
			# Log Epoch, Batch, Loss, LR
			log.info('Train Epoch: {} [{}/{}]\tLoss: {:.3e}\tLR: {:.2e}'
         		.format(epoch, batch_idx, num_batches, loss.item(), scheduler.get_last_lr()[0]))
	train_loss /= num_batches
	log.info('====> Epoch: {} Average loss: {:.3e}'.format(epoch, train_loss))
	writer.add_scalar('Train/Loss', train_loss, epoch)
	writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
	return train_loss

def validate(writer, epoch, val_loader, model, device, criterion):
	model.eval()
	val_loss = 0
	nmse = 0
	num_batches = len(val_loader)
	with torch.no_grad():
		for i, (data,) in enumerate(val_loader):
			data = data.to(device)
			output = model(data)
			val_loss += criterion(output, data)
			nmse += evaluator(output, data)

	val_loss /= num_batches
	nmse /= num_batches

	writer.add_scalar('Validation/Loss', val_loss, epoch)
	writer.add_scalar('Validation/NMSE', nmse, epoch)
	log.info('====> Validation set loss: {:.3e}, NMSE: {:.3e}'.format(val_loss, nmse))
	
	return val_loss, nmse

def test(val_loader, model, device, criterion):
	model.eval()
	val_loss = 0
	nmse = 0
	num_batches = len(val_loader)
	with torch.no_grad():
		for i, (data, _) in enumerate(val_loader):
			data = data.to(device)
			output = model(data)
			val_loss += criterion(output, data)
			nmse += evaluator(output, data)

	val_loss /= num_batches
	nmse /= num_batches
	
	
	log.info('====> Test set loss: {:.3e}, NMSE: {:.3e}'.format(val_loss, nmse))
	return val_loss, nmse

if __name__ == '__main__':
	main()