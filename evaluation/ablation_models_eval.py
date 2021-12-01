import argparse
import parser
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ablation_models import (BarlowTwinsContrastiveTripletNet,
                             BarlowTwinsDBRatioNet, ContrastiveTripletNet,
                             DBRatioContrastiveNet, FullSSLNet)
from datasets import Dataset
from utils import _get_subject_level_split, _map_age, get_patient_prediction

stats_test_data = { }

# features of the models
args_p = dict({
	# data loading
	"batch_size": 4096, # CAUTION: actual batch sent to gpu varies depending on # bad epochs (training) and # windows in subject (inference)
	"num_workers": 16,
	"pin_memory": True, # CAUTION: =True leads to slower data loading on Vivaldi - don't know why.
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

# This function is used to collect final metrics.
def collect_metrics_heldout(y_probs_test, y_true_test, y_pred_test, sample_indices_test, _INDEX_DF, dataset):

	dataset_index = _INDEX_DF
	
	if dataset == "lemon":
		title = "subject_id"
	elif dataset == "tuh":
		title = "patient_id"
	else:
		raise ValueError("unknown dataset")
	# create patient-level train and test dataframes
	rows = [ ]
	for i in range(len(sample_indices_test)):
		idx = sample_indices_test[i]
		temp = { }
		temp[title] = str(dataset_index.loc[idx, title])
		temp["raw_file_path"] = str(dataset_index.loc[idx, "raw_file_path"])
		temp["sample_idx"] = idx
		temp["y_true"] = y_true_test[i]
		temp["y_probs_0"] = y_probs_test[i, 0]
		temp["y_probs_1"] = y_probs_test[i, 1]
		temp["y_pred"] = y_pred_test[i]
		rows.append(temp)
	test_patient_df = pd.DataFrame(rows)

	# get patient-level metrics from window-level dataframes
	y_probs_test_patient, y_true_test_patient, y_pred_test_patient = get_patient_prediction(test_patient_df, dataset)
	auc_patient_history, prec_history, recall_history, f1_history, bacc_history = [], [], [], [], []
	
	for i in tqdm(range(len(y_probs_test_patient))):
		y_prob = np.copy(y_probs_test_patient)
		y_prob = np.delete(y_prob, i, axis=0)

		y_true_s = np.copy(y_true_test_patient)
		y_true_s = np.delete(y_true_s, i)
		
		stats_test_data[f"probs_0"] = y_prob[:, 0]
		stats_test_data[f"probs_1"] = y_prob[:, 1]

		patient_csv_dict = { }

		# PATIENT-LEVEL ROC PLOT - select optimal threshold for this, and get patient-level precision, recall, f1
		fpr, tpr, thresholds = roc_curve(y_true_s, y_prob[:,1], pos_label=1)
		patient_csv_dict[f"fpr_fold"] = fpr
		patient_csv_dict[f"tpr_fold"] = tpr
		patient_csv_dict[f"thres_fold"] = thresholds

		# select an optimal threshold using the ROC curve
		# Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
		optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]

		# calculate class predictions and confusion-based metrics using the optimal threshold
		roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_prob[:,1]]

		precision_patient_test =  precision_score(y_true_s, roc_predictions, pos_label=1)
		recall_patient_test =  recall_score(y_true_s, roc_predictions, pos_label=1)
		f1_patient_test = f1_score(y_true_s, roc_predictions, pos_label=1)
		bal_acc_patient_test = balanced_accuracy_score(y_true_s, roc_predictions)

		# PATIENT-LEVEL AUROC
		auroc_patient_test = roc_auc_score(y_true_s, y_prob[:,1])

		auc_patient_history.append(auroc_patient_test)
		prec_history.append(precision_patient_test)
		recall_history.append(recall_patient_test)
		f1_history.append(f1_patient_test)
		bacc_history.append(bal_acc_patient_test)
	
	return auc_patient_history, prec_history, recall_history, f1_history, bacc_history

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Execute heldout evaluation.")
	parser.add_argument('--gpu_idx', type=int, help="Index of GPU device. Default is 0.")
	parser.add_argument('--task', type=str, help="choose from ...")
	parser.add_argument('--mode', type=str, help="choose a mode")
	parser.add_argument('--dataset', type=str, help="choose between tuh and leomon")
	args = parser.parse_args()
	
	_GPU_IDX = args.gpu_idx
	_MODE = args.mode
	_TASK = args.task
	dataset=args.dataset

	if dataset == "lemon":
		_INDEX_PATH = "../resources/lemon_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
	elif dataset == "tuh":
		_INDEX_PATH = "../resources/abnormal_corpus_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
		_INDEX_DF['parsed_age_binary'] = _INDEX_DF['parsed_age'].map(_map_age)
	else:
		raise ValueError("unknown dataset")

	_RANDOM_SEED = 42

	#Set seed for numpy and torch
	np.random.seed(_RANDOM_SEED)
	torch.manual_seed(_RANDOM_SEED)
	print(f"Numpy and PyTorch seed set to {_RANDOM_SEED} for reproducibility.")

	#set cuda device
	_TORCH_DEVICE = torch.device(f'cuda:{_GPU_IDX}' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(_TORCH_DEVICE)
	torch.cuda.empty_cache()

	#select moodel 
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
		raise ValueError('unknown model!')
	
	# add 512->2 linear layer to backbone
	model = model.to(_TORCH_DEVICE)
	model = model.embedder[0]
	model.fc = torch.nn.Linear(512, 2)
	model.fc.weight.data.normal_(mean=0.0, std=0.01)
	model.fc.bias.data.zero_()

	# load trained model states
	file = f"../resources/finetuned_models/{dataset}_{_TASK}_{_MODE}.ckpt"
	ckpt = torch.load(file, map_location=_TORCH_DEVICE)
	model.load_state_dict(ckpt['model_state'])
	model = model.to(_TORCH_DEVICE)

	# Cross Entropy loss function
	loss_function = torch.nn.CrossEntropyLoss()

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
	
	# map labels to 0/1
	label_mapping, y = np.unique(y, return_inverse=True)
	print(f"Unique labels 0/1 mapping: {label_mapping}")

	# make train, test, heldout splits. Only heldout set is used for evaluation. Random see is used to ensure reproductability.
	all_subjects, train_subjects, val_subjects, heldout_subjects, train_window_idx, val_window_idx, heldout_window_idx = _get_subject_level_split(_INDEX_DF, _RANDOM_SEED,dataset, _TASK)
	print("train_subjects: ", len(train_subjects), " val subjects: ", len(val_subjects), \
		" heldout_subjects: ", len(heldout_subjects), " all subjects: ", len(all_subjects))
	print(f"len train_window_idx: {len(train_window_idx)}, len val_window_idx: {len(val_window_idx)}, len heldout_window_idx:{len(heldout_window_idx)}")


	# This is for lemon dataset, condition task only. 
	if _TASK == "condition" and dataset == "lemon":
		heldout_indices_all = []
		for i in tqdm(range(len(heldout_subjects))):
			heldout_subjects_resample = np.copy(heldout_subjects)
			heldout_subjects_resample = np.delete(heldout_subjects_resample, i)

			heldout_indices = _INDEX_DF.index[_INDEX_DF["subject_id"].astype("str").isin(heldout_subjects_resample)].tolist()
			heldout_indices_all.append(heldout_indices)
		
		# Datasets, dataloaders
		dataset_list, dataloader_list = [],[]

		for i in tqdm(range(len(heldout_indices_all))):
			heldout_dataset = Dataset(window_idx=heldout_indices_all[i],X=X, y=y)
			dataset_list.append(heldout_dataset)
			heldout_loader = DataLoader(heldout_dataset, batch_size=512, shuffle=False, num_workers=8)
			dataloader_list.append(heldout_loader)

		#initialize metrics lists
		all_auc, all_auc_patient, all_prec, all_recall, all_f1, all_bacc = [],[],[],[],[],[]
		# start evaluation process
		for j in tqdm(range(len(heldout_indices_all))):
			model.eval()
			with torch.no_grad():
				y_probs = torch.empty(0, 2).to(_TORCH_DEVICE)
				y_true = [ ]
				y_pred = [ ]
				window_indices = [ ]
				
				for i, batch in enumerate(dataloader_list[j]):
					
					window_indices += batch['window_idx'].tolist()
					X_batch = batch['feature_data'].to(device=_TORCH_DEVICE, non_blocking=True).float()
					y_batch = batch['label'].to(device=_TORCH_DEVICE, non_blocking=True)
					outputs = model(X_batch)

					_, predicted = torch.max(outputs.data, 1)
					y_pred += predicted.cpu().numpy().tolist()

					# concatenate along 0th dimension
					y_probs = torch.cat((y_probs, outputs.data), 0)
					y_true += y_batch.cpu().numpy().tolist()

			# returning prob distribution over target classes, take softmax across the 1st dimension
			y_probs = torch.nn.functional.softmax(y_probs, dim=1).cpu().numpy()
			y_true = np.array(y_true)

			auroc_test = roc_auc_score(y_true, y_probs[:,1])
			bacc_test = balanced_accuracy_score(y_true, y_pred)
			all_auc.append(auroc_test)
			all_bacc.append(bacc_test)

			report = classification_report(y_true, y_pred, output_dict=True)
			all_prec.append(report['1']['precision'])
			all_recall.append(report['1']['recall'])
			all_f1.append(report['1']['f1-score'])
		
		print("Report-------------------------------")
		print(f"heldout test AUROC: {np.mean(all_auc):.3f}({np.std(all_auc):.3f})")
		print(f"heldout test PRECISION: {np.mean(all_prec):.3f}({np.std(all_prec):.3f})")
		print(f"heldout test RECALL: {np.mean(all_recall):.3f}({np.std(all_recall):.3f})")
		print(f"heldout test F-1: {np.mean(all_f1):.3f}({np.std(all_f1):.3f})")
		print(f"heldout test BALANCED ACCURACY: {np.mean(all_bacc):.3f}({np.std(all_bacc):.3f})")

		# calculate interval percentile
		alpha = 0.95
		p = ((1.0-alpha)/2.0) * 100
		lower = max(0.0, np.percentile(all_auc, p))
		p = (alpha+((1.0-alpha)/2.0)) * 100
		upper = min(1.0, np.percentile(all_auc, p))
	
		# confidence interval
		print('%.1f%% AUC confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
		print("[MAIN] exiting...")
		sys.exit()
	

	# leave-1-out sampling for other tasks
	heldout_dataset = Dataset(window_idx=heldout_window_idx, X=X, y=y)
	heldout_loader = DataLoader(heldout_dataset, batch_size=4096, shuffle=False, num_workers=8)

	# evaluation
	model.eval()
	with torch.no_grad():
		y_probs = torch.empty(0, 2).to(_TORCH_DEVICE)
		y_true = [ ]
		y_pred = [ ]
		window_indices = [ ]
		for i, batch in enumerate(heldout_loader):
			
			window_indices += batch['window_idx'].tolist()
			X_batch = batch['feature_data'].to(device=_TORCH_DEVICE, non_blocking=True).float()
			y_batch = batch['label'].to(device=_TORCH_DEVICE, non_blocking=True)
			outputs = model(X_batch)

			_, predicted = torch.max(outputs.data, 1)
			y_pred += predicted.cpu().numpy().tolist()

			# concatenate along 0th dimension
			y_probs = torch.cat((y_probs, outputs.data), 0)
			y_true += y_batch.cpu().numpy().tolist()

		# returning prob distribution over target classes, take softmax across the 1st dimension
		y_probs = torch.nn.functional.softmax(y_probs, dim=1).cpu().numpy()
		y_true = np.array(y_true)

		auc_patient_history, prec_history, recall_history, f1_history, bacc_history = collect_metrics_heldout(y_probs_test=y_probs,
							y_true_test=y_true,
							y_pred_test=y_pred,
							dataset=dataset,
							sample_indices_test = window_indices, _INDEX_DF=_INDEX_DF)
	
	# confidence interval
	alpha = 0.95

	print("Report-------------------------------")
	print(f"heldout test patient AUROC: {np.mean(auc_patient_history):.3f}({np.std(auc_patient_history):.4f})")
	print(f"heldout test patient PRECISION: {np.mean(prec_history):.3f}({np.std(prec_history):.4f})")
	print(f"heldout test patient RECALL: {np.mean(recall_history):.3f}({np.std(recall_history):.4f})")
	print(f"heldout test patient F-1: {np.mean(f1_history):.3f}({np.std(f1_history):.4f})")
	print(f"heldout test patient BALANCED ACCURACY: {np.mean(bacc_history):.3f}({np.std(bacc_history):.4f})")

	# calculate interval percentile
	p = ((1.0-alpha)/2.0) * 100
	lower = max(0.0, np.percentile(auc_patient_history, p))
	p = (alpha+((1.0-alpha)/2.0)) * 100
	upper = min(1.0, np.percentile(auc_patient_history, p))
	
	# confidence interval
	print('%.1f%% AUC confidence interval %.2f%% and %.2f%%' % (alpha*100, lower*100, upper*100))

	print("[MAIN] exiting...")
