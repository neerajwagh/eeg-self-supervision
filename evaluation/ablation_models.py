import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class FullSSLNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args['projector'].split('-')[-1])

		# core embedding network
		self.embedder = self._build_embedder()

		# task-specific "heads"
		self.barlow_head = self._build_barlow_bn_head()
		self.dbr_head_left, self.dbr_head_right = self._build_dbr_heads()

		return

	def forward(self, x):
		z = self.embedder[0](x)
		return z

	def _build_embedder(self):
		layers = []
		
		# encoder network
		encoder = torchvision.models.resnet18(zero_init_residual=True)
		# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		encoder.conv1 = nn.Conv2d(in_channels=self.args["input_channels"], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		encoder.fc = nn.Identity()
		layers.append(encoder)
		
		# projector network
		sizes = [512] + list(map(int, self.args["projector"].split('-')))
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		
		embedder = nn.Sequential(*layers)
		return embedder

	def _build_barlow_bn_head(self):
		return nn.BatchNorm1d(self.embedding_dim, affine=False, track_running_stats=False)

	def _build_dbr_heads(self):
		linear_left = nn.Linear(self.embedding_dim, 1, bias=False)
		linear_right = nn.Linear(self.embedding_dim, 1, bias=False)
		return linear_left, linear_right

class ContrastiveTripletNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args['projector'].split('-')[-1])

		# core embedding network
		self.embedder = self._build_embedder()

		self.soft_margin_loss = nn.SoftMarginLoss()

		self.triplet_margin_loss = nn.TripletMarginLoss(
			p=args["loss_norm"],
			margin=args['loss_margin'],
		)

	def embed(self, x):
		z = self.embedder(x)
		return z

	def forward_triplet(self, x_anchor, x_pos, x_neg, return_embeddings=False, return_loss=False):
		
		z_anchor = self.embedder(x_anchor)
		z_pos = self.embedder(x_pos)
		z_neg = self.embedder(x_neg)

		if return_embeddings:
			return z_anchor, z_pos, z_neg

		if return_loss:
			loss = self.triplet_margin_loss(z_anchor, z_pos, z_neg)
			return loss


	def _build_embedder(self):
		layers = []
		
		# encoder network
		encoder = torchvision.models.resnet18(zero_init_residual=True)
		# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		encoder.conv1 = nn.Conv2d(in_channels=self.args["input_channels"], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

		encoder.fc = nn.Identity()
		layers.append(encoder)
		
		# projector network
		sizes = [512] + list(map(int, self.args["projector"].split('-')))
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

		embedder = nn.Sequential(*layers)
		return embedder

# https://github.com/facebookresearch/barlowtwins/blob/6f91e07150654ec1eb233cedb3b826ba0b32589c/main.py
class BarlowTwinsDBRatioNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args['projector'].split('-')[-1])

		# core embedding network
		self.embedder = self._build_embedder()

		# task-specific "heads"
		self.barlow_bn_head = self._build_barlow_bn_head()
		self.dbr_head_left, self.dbr_head_right = self._build_dbr_heads()
		return 

	def _build_embedder(self):
		layers = []
		
		# encoder network
		encoder = torchvision.models.resnet18(zero_init_residual=True)
		# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		encoder.conv1 = nn.Conv2d(in_channels=self.args["input_channels"], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		encoder.fc = nn.Identity()
		layers.append(encoder)
		
		# projector network
		sizes = [512] + list(map(int, self.args["projector"].split('-')))
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

		embedder = nn.Sequential(*layers)
		return embedder

	def embed(self, x):
		z = self.embedder(x)
		return z

	def get_dbr_prediction(self, x):
		z = self.embedder(x)
		y1 = F.relu(self.dbr_head_left(z))
		y2 = F.relu(self.dbr_head_right(z))
		y_pred_concat = torch.cat((y1, y2), dim=1)
		y_pred_avg = torch.mean(y_pred_concat, dim=1, keepdims=False)
		return y_pred_avg

	def _build_barlow_bn_head(self):
		return nn.BatchNorm1d(self.embedding_dim, affine=False, track_running_stats=False)

	def _build_dbr_heads(self):
		linear_left = nn.Linear(self.embedding_dim, 1, bias=False)
		linear_right = nn.Linear(self.embedding_dim, 1, bias=False)
		return linear_left, linear_right

	# CAUTION: during training - batch size is variable depending on how many bad epochs/windows are dropped in the loaded batch
	# CAUTION: during inference - batch size is variable depending on how many bad epochs/windows are present in input recordings
	def forward(self, x_left, x_right, y_left, y_right):

		batch_size = x_left.shape[0]

		z1 = self.embedder(x_left)
		z2 = self.embedder(x_right)

		# empirical cross-correlation matrix
		bn_z1 = self.barlow_bn_head(z1)
		bn_z2 = self.barlow_bn_head(z2)

		y1 = F.relu(self.dbr_head_left(z1))
		y2 = F.relu(self.dbr_head_right(z2))

		ccr = bn_z1.T @ bn_z2
		ccr.div_(batch_size)

		on_diag = torch.diagonal(ccr).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(ccr).pow_(2).sum()
		barlow_twins_loss = on_diag + (self.args["lambda"] * off_diag)

		y1 = torch.squeeze(y1, dim=1)
		y2 = torch.squeeze(y2, dim=1)

		y_left = torch.unsqueeze(y_left, dim=1)
		y_right = torch.unsqueeze(y_right, dim=1)
		y_true_concat = torch.cat((y_left, y_right), dim=1)
		y_true_avg = torch.mean(y_true_concat, dim=1, keepdims=False)
		left_mse_loss = F.mse_loss(y1, y_true_avg)
		right_mse_loss = F.mse_loss(y2, y_true_avg)

		return barlow_twins_loss, left_mse_loss, right_mse_loss


class BarlowTwinsContrastiveTripletNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args['projector'].split('-')[-1])

		# core embedding network
		self.embedder = self._build_embedder()

		# task-specific "heads"
		self.barlow_bn_head = self._build_barlow_bn_head()
		
		return 

	def embed(self, x):
		z = self.embedder(x)
		return z

	def _dbr_forward(self, x, y_c3, y_c4):
		dbr_true = y_c3 + y_c4 / 2.0
		z = self.embedder(x)
		dbr_pred = F.relu(self.dbr_linear(z))
		dbr_pred = torch.squeeze(dbr_pred, dim=1)
		mse_loss = F.mse_loss(dbr_true, dbr_pred)
		return mse_loss

	def _build_embedder(self):
		layers = []
		
		# encoder network
		encoder = torchvision.models.resnet18(zero_init_residual=True)
		# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		encoder.conv1 = nn.Conv2d(in_channels=self.args["input_channels"], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		encoder.fc = nn.Identity()
		layers.append(encoder)
		
		# projector network
		sizes = [512] + list(map(int, self.args["projector"].split('-')))
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		
		embedder = nn.Sequential(*layers)
		return embedder


	def _build_barlow_bn_head(self):
		return nn.BatchNorm1d(self.embedding_dim, affine=False, track_running_stats=False)


	def _forward_barlow(self, x_left, x_right, return_embeddings=False, return_batch_ccr=False, return_input_ccr=False, return_loss=False):
		batch_size = x_left.shape[0]
		
		z1 = self.embedder(x_left)
		z2 = self.embedder(x_right)

		bn_z1 = self.barlow_bn_head(z1)
		bn_z2 = self.barlow_bn_head(z2)

		if return_embeddings:
			return bn_z1, bn_z2
		
		if return_batch_ccr:
			embeddings_ccr = bn_z1.T @ bn_z2
			embeddings_ccr.div_(batch_size)
			return embeddings_ccr

		if return_input_ccr:

			# flatten input to shape: batch size x flattened spect features
			# feat_dim = self.args["input_channels"] * 38 * 40
			feat_dim = self.args["input_channels"] * 32 * 32
			x_left = x_left.reshape((batch_size, feat_dim))
			x_right = x_right.reshape((batch_size, feat_dim))

			# normalize each feature independently - across batch
			x_left = F.normalize(x_left, dim=0)
			x_right = F.normalize(x_right, dim=0)

			# compute ccr
			print(x_left.shape, x_right.shape)
			input_ccr = x_left.T @ x_right
			input_ccr.div_(batch_size)
			return input_ccr

		if return_loss:
			ccr = bn_z1.T @ bn_z2
			ccr.div_(batch_size)

			on_diag = torch.diagonal(ccr).add_(-1).pow_(2).sum()
			off_diag = off_diagonal(ccr).pow_(2).sum()
			barlow_twins_loss = on_diag + (self.args["lambda"] * off_diag)

			return barlow_twins_loss

class DBRatioContrastiveNet(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.embedding_dim = int(args['projector'].split('-')[-1])

		# core embedding network
		self.embedder = self._build_embedder()

		# task-specific "heads"
		self.dbr_linear = self._build_dbr_head()

	def embed(self, x):
		z = self.embedder(x)
		return z

	def get_dbr_prediction(self, x):
		z = self.embedder(x)
		y1 = F.relu(self.dbr_linear(z))
		y_pred_avg = torch.squeeze(y1, dim=1)
		return y_pred_avg

	def _dbr_forward(self, x, y_c3, y_c4):
		dbr_true = y_c3 + y_c4 / 2.0
		z = self.embedder(x)
		dbr_pred = F.relu(self.dbr_linear(z))
		dbr_pred = torch.squeeze(dbr_pred, dim=1)
		mse_loss = F.mse_loss(dbr_true, dbr_pred)
		return mse_loss

	def _build_embedder(self):
		layers = []
		
		# encoder network
		encoder = torchvision.models.resnet18(zero_init_residual=True)
		# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
		encoder.conv1 = nn.Conv2d(in_channels=self.args["input_channels"], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		encoder.fc = nn.Identity()
		layers.append(encoder)
		
		# projector network
		sizes = [512] + list(map(int, self.args["projector"].split('-')))
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

		embedder = nn.Sequential(*layers)
		return embedder

	def _build_dbr_head(self):
		dbr_linear = nn.Linear(self.embedding_dim, 1, bias=False)
		return dbr_linear

def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
