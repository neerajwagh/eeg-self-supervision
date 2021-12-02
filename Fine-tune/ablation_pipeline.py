import os
import numpy as np
import pandas as pd
import torch
import argparse
import parser
import torch.nn.functional as F

from ablation_models import BarlowTwinsDBRatioNet, ContrastiveTripletNet, DBRatioContrastiveNet, FullSSLNet, BarlowTwinsContrastiveTripletNet
from torch.utils.data import WeightedRandomSampler, DataLoader
from datasets import Dataset
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import _map_age, _get_subject_level_split, get_patient_prediction


# Metrics array
train_auc_history = []
valid_auc_history = []

train_precision_history = []
valid_precision_history = []

train_recall_history = []
valid_recall_history = []

train_f1_history = []
valid_f1_history = []

train_bacc_history = []
valid_bacc_history = []

train_loss_history = []
valid_loss_history = []

#patient-level metrics
train_auc_history_patient = []
valid_auc_history_patient = []
	
train_precision_history_patient = []
valid_precision_history_patient = []
	
train_recall_history_patient = []
valid_recall_history_patient = []

train_f1_history_patient = []
valid_f1_history_patient = []

train_bacc_history_patient = []
valid_bacc_history_patient = []

stats_test_data = { }
# after each epoch, record all the metrics on both train and validation sets
def collect_metrics(y_probs_train, y_true_train, y_pred_train, sample_indices_train,
					y_probs_valid, y_true_valid, y_pred_valid, sample_indices_valid,
					train_loss, valid_loss, dataset):

	if dataset == "lemon":
		title = "subject_id"
	elif dataset == "tuh":
		title = "patient_id"
	else:
		raise ValueError("unknown dataset")

	# create patient-level train and test dataframes
	rows = [ ]
	for i in range(len(sample_indices_train)):
		idx = sample_indices_train[i]
		temp = { }

		temp[title] = str(_INDEX_DF.loc[idx, title])
		temp["raw_file_path"]=str(_INDEX_DF.loc[idx, "raw_file_path"])
		temp["sample_idx"] = idx
		temp["y_true"] = y_true_train[i]
		temp["y_probs_0"] = y_probs_train[i, 0]
		temp["y_probs_1"] = y_probs_train[i, 1]
		temp["y_pred"] = y_pred_train[i]
		rows.append(temp)
	train_patient_df = pd.DataFrame(rows)

	rows = [ ]
	for i in range(len(sample_indices_valid)):
		idx = sample_indices_valid[i]
		temp = { }	

		temp[title] = str(_INDEX_DF.loc[idx, title])
		temp["raw_file_path"]=str(_INDEX_DF.loc[idx, "raw_file_path"])
		temp["sample_idx"] = idx
		temp["y_true"] = y_true_valid[i]
		temp["y_probs_0"] = y_probs_valid[i, 0]
		temp["y_probs_1"] = y_probs_valid[i, 1]
		temp["y_pred"] = y_pred_valid[i]
		rows.append(temp)
	test_patient_df = pd.DataFrame(rows)

	if  (( _TASK == "condition") and (dataset=="lemon") ) == False:
		# get patient-level metrics from window-level dataframes
		y_probs_train_patient, y_true_train_patient, y_pred_train_patient = get_patient_prediction(train_patient_df, dataset)
		y_probs_valid_patient, y_true_valid_patient, y_pred_valid_patient = get_patient_prediction(test_patient_df, dataset)

		# PATIENT-LEVEL AUROC
		train_auc_history_patient.append(roc_auc_score(y_true_train_patient, y_probs_train_patient[:,1]))
		valid_auc_history_patient.append(roc_auc_score(y_true_valid_patient, y_probs_valid_patient[:,1]))

		# PATIENT-LEVEL PRECISION
		train_precision_history_patient.append(precision_score(y_true_train_patient, y_pred_train_patient))
		valid_precision_history_patient.append(precision_score(y_true_valid_patient, y_pred_valid_patient))

		# PATIENT-LEVEL RECALL
		train_recall_history_patient.append(recall_score(y_true_train_patient, y_pred_train_patient))
		valid_recall_history_patient.append(recall_score(y_true_valid_patient, y_pred_valid_patient))

		# PATIENT-LEVEL F-1
		train_f1_history_patient.append(f1_score(y_true_train_patient, y_pred_train_patient))
		valid_f1_history_patient.append(f1_score(y_true_valid_patient, y_pred_valid_patient))

		# PATIENT-LEVEL BACC
		train_bacc_history_patient.append(balanced_accuracy_score(y_true_train_patient, y_pred_train_patient))
		valid_bacc_history_patient.append(balanced_accuracy_score(y_true_valid_patient, y_pred_valid_patient))
	# LOSS - epoch loss is defined as mean of minibatch losses within epoch 
	train_loss_history.append(np.mean(train_loss))
	valid_loss_history.append(np.mean(valid_loss))

	# CAUTION - The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	train_auc_history.append(roc_auc_score(y_true_train, y_probs_train[:,1]))
	valid_auc_history.append(roc_auc_score(y_true_valid, y_probs_valid[:,1]))

	# PRECISION
	train_precision_history.append(precision_score(y_true_train, y_pred_train))
	valid_precision_history.append(precision_score(y_true_valid, y_pred_valid))

	# RECALL
	train_recall_history.append(recall_score(y_true_train, y_pred_train))
	valid_recall_history.append(recall_score(y_true_valid, y_pred_valid))

	# F-1
	train_f1_history.append(f1_score(y_true_train, y_pred_train))
	valid_f1_history.append(f1_score(y_true_valid, y_pred_valid))

	# balanced_acc
	train_bacc_history.append(balanced_accuracy_score(y_true_train, y_pred_train))
	valid_bacc_history.append(balanced_accuracy_score(y_true_valid, y_pred_valid))

	# http://kitchingroup.cheme.cmu.edu/blog/2013/01/21/Controlling-the-format-of-printed-variables/
	print("[TRAIN] loss: {0:1.3f}, auroc: {1:1.3f}, f-1: {2:1.3f}, precision: {3:1.3f}, recall: {4:1.3f}, bacc: {5:1.3f}".format(
		train_loss_history[-1],
		train_auc_history[-1],
		train_f1_history[-1],
		train_precision_history[-1],
		train_recall_history[-1],
		train_bacc_history[-1]
	))

	print("[VALID] loss: {0:1.3f}, auroc: {1:1.3f}, f-1: {2:1.3f}, precision: {3:1.3f}, recall: {4:1.3f}, bacc: {5:1.3f}".format(
		valid_loss_history[-1],
		valid_auc_history[-1],
		valid_f1_history[-1],
		valid_precision_history[-1],
		valid_recall_history[-1],
		valid_bacc_history[-1]
	))
	if (( _TASK == "condition") and (dataset=="lemon") ) == False:
		print("[PATIENT-TRAIN] auroc: {0:1.3f}, f-1: {1:1.3f}, precision: {2:1.3f}, recall: {3:1.3f}, bal_acc: {4:1.3f}".format(
			train_auc_history_patient[-1],
			train_f1_history_patient[-1],
			train_precision_history_patient[-1],
			train_recall_history_patient[-1],
			train_bacc_history_patient[-1]
		))

		print("[PATIENT-VALID] auroc: {0:1.3f}, f-1: {1:1.3f}, precision: {2:1.3f}, recall: {3:1.3f}, bal_acc:{4:1.3f}".format(
			valid_auc_history_patient[-1],
			valid_f1_history_patient[-1],
			valid_precision_history_patient[-1],
			valid_recall_history_patient[-1],
			valid_bacc_history_patient[-1]
		))

	return


args_p = dict({
	# data loading
	"batch_size": 4096, # CAUTION: actual batch sent to gpu varies depending on # bad epochs (training) and # windows in subject (inference)
	"num_workers": 16,
	"pin_memory": False, # CAUTION: =True leads to slower data loading on Vivaldi - don't know why.
	"prefetch_factor": 2,
	"persistent_workers": False, # =True crashs after first epoch
	# model architecture and loss function
	"feature_type": 'topo',
	"input_channels": 7,
	"flattened_features_per_sample": 7*32*32,
	"projector": '512-512-512',
	"loss_margin": 0.2, # ignored for eval
	"loss_norm": 2, # ignored for eval
	# for BT loss
	"lambda": 2e-3,
})

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Execute evaluation pipeline for contrastive_triplet.")
	parser.add_argument('--gpu_idx', type=int, default=0, help="Index of GPU device. Default is 0.")
	parser.add_argument('--task', type=str, default="gender", help="choose from ...")
	parser.add_argument('--mode', type=str, default="FULLSSL", help="choose a mode")
	parser.add_argument('--dataset', type=str, default="lemon", help="choose lemon or tuh...")
	args = parser.parse_args()

	_GPU_IDX = args.gpu_idx
	_TASK = args.task
	_MODE = args.mode
	dataset = args.dataset

	_EPOCH = 1511
	lr = 0.002
	
	# creat directories to store checkpoints
	if not os.path.exists(f"{dataset}_{_MODE}"):
		os.makedirs(f"{dataset}_{_MODE}", exist_ok=True)

	# ensure reproducibility of results
	_RANDOM_SEED = 42
	np.random.seed(_RANDOM_SEED)
	print(f"Numpy seed set to {_RANDOM_SEED} for reproducibility.")

	torch.backends.cudnn.enabled = True
	print("cuDNN backend enabled.")

	# Set cuda device
	_TORCH_DEVICE = torch.device(f'cuda:{_GPU_IDX}' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(_TORCH_DEVICE)
	torch.cuda.empty_cache()
	print(f'Using device: {_TORCH_DEVICE}, {torch.cuda.get_device_name(_TORCH_DEVICE)}')

	# initialize abllition model
	if _MODE == "FULLSSL":
		model = FullSSLNet(args_p)
	elif _MODE == "DBRTriplet":
		model = DBRatioContrastiveNet(args_p)
	elif _MODE == "BTTriplet":
		model = BarlowTwinsContrastiveTripletNet(args_p)
	elif _MODE == "BTDBR":
		model = BarlowTwinsDBRatioNet(args_p)
	elif _MODE == "Triplet":
		model = ContrastiveTripletNet(args_p)
	elif _MODE == "DBR":
		model = BarlowTwinsDBRatioNet(args_p)
	elif _MODE == "BT":
		model = FullSSLNet(args_p)
	else:
		raise ValueError('unknown mode!')

	# load pretrained states
	ckpt = torch.load(f"../resources/pretrained_models/pretrained_{_MODE}.ckpt", map_location=_TORCH_DEVICE)
	model = model.to(_TORCH_DEVICE)
	model.load_state_dict(ckpt['model_state'])
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)

	# load pretrained optimizer
	optimizer.load_state_dict(ckpt['optimizer_state'])
	for param_group in optimizer.param_groups:
	 	param_group['lr'] = lr

	# create the classification layer
	model = model.embedder[0]
	model.fc = torch.nn.Linear(512, 2)
	model.fc.weight.data.normal_(mean=0.0, std=0.01)
	model.fc.bias.data.zero_()

	optimizer.add_param_group({'params': model.fc.parameters()})
	model = model.to(_TORCH_DEVICE)

	# Fine-tune
	model.requires_grad_(True)
	model.fc.requires_grad_(True)

	loss_function = torch.nn.CrossEntropyLoss()

		# load dataset indices
	if dataset == "lemon":
		_INDEX_PATH = "../resources/lemon_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
	elif dataset == "tuh":
		_INDEX_PATH = "../resources/abnormal_corpus_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
		_INDEX_DF['parsed_age_binary'] = _INDEX_DF['parsed_age'].map(_map_age)
	else:
		raise ValueError("unknown dataset")

	#load X, y
	print("Loading data...")
	X = np.load(f'../resources/topo_data_{dataset}.npy', mmap_mode='r').astype(np.float32)		
	print("Loading done.")

	if dataset == 'tuh':
		if _TASK == "gender":
			y = _INDEX_DF['parsed_gender'].to_numpy()
			_INDEX_DF = _INDEX_DF.loc[_INDEX_DF["parsed_gender"].isin(["Male", "Female"])]
		elif _TASK == "age":
			y = _INDEX_DF['parsed_age_binary'].to_numpy()
			_INDEX_DF = _INDEX_DF.loc[_INDEX_DF["parsed_age_binary"].isin(["young", "old"])]
		elif _TASK == "condition":
			y = _INDEX_DF["text_label"].to_numpy()
			_INDEX_DF = _INDEX_DF.loc[_INDEX_DF["text_label"].isin(["normal", "abnormal"])]
		else:
			raise ValueError('unknown task!')

		keep_idx = _INDEX_DF.index
		X = X[keep_idx, ...]
		y = y[keep_idx, ...]
		_INDEX_DF.reset_index(drop=True, inplace=True)
		_INDEX_DF.sort_index(inplace=True)
		_INDEX_DF['new_window_idx'] = _INDEX_DF.index
	
	elif dataset == "lemon":
		if _TASK == "gender":
			y = _INDEX_DF['gender'].to_numpy()
		elif _TASK == "age":
			y = _INDEX_DF['age'].to_numpy()
		elif _TASK == "condition":
			y = _INDEX_DF["condition"].to_numpy()
		else:
			raise ValueError('unknown task!')
	else:
		raise ValueError('unknown dataset!')
	
	label_mapping, y = np.unique(y, return_inverse=True)
	print(f"Unique labels 0/1 mapping: {label_mapping}")

	# Patient level subject split, 30% for heldout, 49% for train, 21% for validation
	all_subjects, train_subjects, val_subjects, heldout_subjects, train_window_idx, val_window_idx, heldout_window_idx = \
	_get_subject_level_split(_INDEX_DF, _RANDOM_SEED, dataset, _TASK)
	all_window_idx = np.concatenate((train_window_idx, val_window_idx, heldout_window_idx), axis=0)
	
	train_and_val_subjects = np.concatenate((train_subjects, val_subjects), axis=0)
	train_and_val_window_idx = np.concatenate((train_window_idx, val_window_idx), axis=0)

	print("train_subjects: ", len(train_subjects), " val subjects: ", len(val_subjects), \
		" heldout_subjects: ", len(heldout_subjects), " all subjects: ", len(all_subjects))
	print(f"len train_window_idx: {len(train_window_idx)}, len val_window_idx: {len(val_window_idx)}, len heldout_window_idx:{len(heldout_window_idx)}")


	# Weighted Random Sampler
	labels_unique, counts = np.unique(y[train_and_val_window_idx], return_counts=True)
	
	class_weights = np.array([1.0 / x for x in counts])
	# provide weights for samples in the training set only
	sample_weights = class_weights[y[train_window_idx]]
	# sampler needs to come up with training set size number of samples
	weighted_sampler = WeightedRandomSampler(
		weights=sample_weights,
		num_samples=len(train_window_idx), replacement=True
	)
	
	# Dataset 
	train_dataset = Dataset(window_idx=train_window_idx, X=X, y=y)
	valid_dataset = Dataset(window_idx=val_window_idx, X=X, y=y)


	# data loader
	train_loader = DataLoader(train_dataset, batch_size=512, num_workers=8, sampler=weighted_sampler)
	valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False,  num_workers=8)
	train_metrics_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)

	# Training
	for epoch in range(_EPOCH):
		model.train()
		train_loss = []

		for batch_idx, batch in enumerate(train_loader):
			# send batch to GPU
			X_batch = batch['feature_data'].to(device=_TORCH_DEVICE, non_blocking=True)
			y_batch = batch['label'].to(device=_TORCH_DEVICE, non_blocking=True)
			optimizer.zero_grad()

			# forward pass
			outputs = model(X_batch)
			loss = loss_function(outputs, y_batch)
			train_loss.append(loss.item())

			# backward pass
			loss.backward()
			optimizer.step()


		#Evaluation 
		model.eval()
		with torch.no_grad():
			y_probs_train = torch.empty(0, 2).to(_TORCH_DEVICE)
			window_idx_train, y_true_train, y_pred_train = [], [], []

			for i, batch in enumerate(train_metrics_loader):

				window_idx_train += batch['window_idx'].tolist()
				X_batch = batch['feature_data'].to(device=_TORCH_DEVICE, non_blocking=True).float()
				y_batch = batch['label'].to(device=_TORCH_DEVICE, non_blocking=True).float()

				# forward pass
				outputs = model(X_batch)

				_, predicted = torch.max(outputs.data, 1)
				y_pred_train += predicted.cpu().numpy().tolist()
				# concatenate along 0th dimension
				y_probs_train = torch.cat((y_probs_train, outputs), 0)
				y_true_train += y_batch.cpu().numpy().tolist()

		# returning prob distribution over target classes, take softmax over the 1st dimension
		y_probs_train = F.softmax(y_probs_train, dim=1).cpu().numpy()
		y_true_train = np.array(y_true_train)

		# evaluate model after each epoch for validation data
		model.eval()
		with torch.no_grad():
			y_probs_valid = torch.empty(0, 2).to(_TORCH_DEVICE)
			window_idx_valid, y_true_valid, valid_loss, y_pred_valid = [], [], [], []

			for i, batch in enumerate(valid_loader):
				window_idx_valid += batch['window_idx'].tolist()
				X_batch = batch['feature_data'].to(device=_TORCH_DEVICE, non_blocking=True).float()
				y_batch = batch['label'].to(device=_TORCH_DEVICE, non_blocking=True)

				# forward pass
				outputs = model(X_batch)

				loss = loss_function(outputs, y_batch)
				valid_loss.append(loss.item())

				_, predicted = torch.max(outputs.data, 1)
				y_pred_valid += predicted.cpu().numpy().tolist()
				# concatenate along 0th dimension
				y_probs_valid = torch.cat((y_probs_valid, outputs.data), 0)
				y_true_valid += y_batch.cpu().numpy().tolist()

		# returning prob distribution over target classes, take softmax over the 1st dimension
		y_probs_valid = F.softmax(y_probs_valid, dim=1).cpu().numpy()
		y_true_valid = np.array(y_true_valid)

		print("Epoch: ", epoch)
		collect_metrics(y_probs_train, y_true_train, y_pred_train, window_idx_train,
						y_probs_valid, y_true_valid, y_pred_valid, window_idx_valid, 
						train_loss, valid_loss, dataset)
		
		state = {
				'epochs': epoch,
				'model_description' : str(model),
				'model_state': model.state_dict(),
				'optimizer': optimizer.state_dict()
				}
		if epoch % 10 == 0:
			torch.save(state, f"{dataset}_{_MODE}/_Epoch_{epoch}.ckpt")
		