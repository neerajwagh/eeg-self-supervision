import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

from utils import _get_subject_level_split, _map_age, get_patient_prediction

stats_test_data = {}

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
	
	# perform leave-1-out sampling
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

def _eval_downstream_task(index_df, dataset, task):

	X = np.load(f'psd_bandpower_relative_{dataset}.npy').astype(np.float32)
	X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

	#load data index file
	if dataset == 'tuh':
		if task == "gender":
			y = index_df['parsed_gender'].to_numpy()
			index_df = index_df.loc[index_df["parsed_gender"].isin(["Male", "Female"])]
		elif task == "age":
			y = index_df['parsed_age_binary'].to_numpy()
			index_df = index_df.loc[index_df["parsed_age_binary"].isin(["young", "old"])]
		elif task == "condition":
			y = index_df["text_label"].to_numpy()
			index_df = index_df.loc[index_df["text_label"].isin(["normal", "abnormal"])]
		else:
			raise ValueError('unknown task!')

		keep_idx = index_df.index
		X = X[keep_idx, ...]
		y = y[keep_idx, ...]
		index_df.reset_index(drop=True, inplace=True)
		index_df.sort_index(inplace=True)
		index_df['new_window_idx'] = index_df.index
	
	elif dataset == "lemon":
		if task == "gender":
			y = index_df['gender'].to_numpy()
			print(y.shape)
		elif task == "age":
			y = index_df['age'].to_numpy()
		elif task == "condition":
			y = index_df["condition"].to_numpy()
		else:
			raise ValueError('unknown task!')
	else:
		raise ValueError('unknown dataset!')

	all_subjects, train_subjects, val_subjects, heldout_subjects, train_window_idx, val_window_idx, heldout_window_idx = \
	_get_subject_level_split(index_df, _RANDOM_SEED, dataset, task)

	# map labels to 0/1
	label_mapping, y = np.unique(y, return_inverse=True)
	print(f"Unique labels 0/1 mapping: {label_mapping}")

	print("heldout_subjects: ", len(heldout_subjects))
	print(f"len heldout_window_idx:{len(heldout_window_idx)}")
	
	#load corresponding model
	final_model = joblib.load(f'models/{dataset}_{task}_linear.pkl', mmap_mode='r')

	heldout_indices_all = []
	X_list, y_list = [], []


	# Leave 1 out sampling for lemon dataset and condition task. 
	if (dataset == "lemon") and (task=="condition"): 
		# bootstrap
		for i in tqdm(range(len(heldout_subjects))):
			heldout_subjects_resample = np.copy(heldout_subjects)
			heldout_subjects_resample = np.delete(heldout_subjects_resample, i)
			
			if i == 0:
				print("length of bootstrap subject: ", len(heldout_subjects_resample))

			# create X and y corresponding to resampled subjects
			heldout_indices = _INDEX_DF.index[_INDEX_DF["subject_id"].astype("str").isin(heldout_subjects_resample)].tolist()
			heldout_indices_all.append(heldout_indices)
			X_heldout = X[heldout_indices, ...]
			y_heldout = y[heldout_indices, ...]
			X_list.append(X_heldout)
			y_list.append(y_heldout)
		print("length of bootstrap indices 0: ", len(heldout_indices_all[0]))


		#initialize metrics lists
		all_auc, all_prec, all_recall, all_f1, all_bacc = [],[],[],[],[]

		# report final metrics on heldout test set
		for j in tqdm(range(len(X_list))):
			X_heldout = X_list[j]
			y_heldout = y_list[j]

			y_pred_heldout = final_model.predict(X_heldout)
			y_probs_heldout = final_model.predict_proba(X_heldout)
			
			auroc_test = roc_auc_score(y_heldout, final_model.predict_proba(X_heldout)[:,1])
			if _TASK == 'condition':
				bal_acc_test = balanced_accuracy_score(y_heldout, y_pred_heldout)
				report = classification_report(y_heldout, y_pred_heldout, output_dict=True)
				all_auc.append(auroc_test)
				all_bacc.append(bal_acc_test)
				all_prec.append(report['1']['precision'])
				all_recall.append(report['1']['recall'])
				all_f1.append(report['1']['f1-score'])

		alpha = 0.95
		print("Report-------------------------------")
		print(f"heldout test AUROC: {np.mean(all_auc):.3f}({np.std(all_auc):.3f})")
		print(f"heldout test PRECISION: {np.mean(all_prec):.3f}({np.std(all_prec):.3f})")
		print(f"heldout test RECALL: {np.mean(all_recall):.3f}({np.std(all_recall):.3f})")
		print(f"heldout test F-1: {np.mean(all_f1):.3f}({np.std(all_f1):.3f})")
		print(f"heldout test BALANCED ACCURACY: {np.mean(all_bacc):.3f}({np.std(all_bacc):.3f})")

		# calculate interval percentile
		p = ((1.0-alpha)/2.0) * 100
		lower = max(0.0, np.percentile(all_auc, p))
		p = (alpha+((1.0-alpha)/2.0)) * 100
		upper = min(1.0, np.percentile(all_auc, p))
		
		# confidence interval
		print('%.1f%% AUC confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

		print("[MAIN] exiting...")
		return

	X_heldout = X[heldout_window_idx, ...]
	y_heldout = y[heldout_window_idx, ...]

	# make predictions on heldout subjects
	y_pred_heldout = final_model.predict(X_heldout)
	y_probs_heldout = final_model.predict_proba(X_heldout)

	# calculate metrics with leave-1-out sampling
	auc_patient_history, prec_history, recall_history, f1_history, bacc_history = collect_metrics_heldout(y_probs_test=y_probs_heldout,
							y_true_test=y_heldout,
							y_pred_test=y_pred_heldout,
							sample_indices_test = heldout_window_idx, _INDEX_DF=index_df, dataset=dataset)
	
	# confidence level
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
	return

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Execute evaluation pipeline using baseline PSD features.")
	parser.add_argument('--dataset', type=str, help="choose between tuh and lemon")
	parser.add_argument('--task', type=str, help="choose from ...")
	args = parser.parse_args()

	dataset = args.dataset
	_TASK = args.task

	# load dataset indices
	if dataset == "lemon":
		_INDEX_PATH = "lemon_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
	elif dataset == "tuh":
		_INDEX_PATH = "abnormal_corpus_window_index.csv"
		_INDEX_DF = pd.read_csv(_INDEX_PATH)
		_INDEX_DF['parsed_age_binary'] = _INDEX_DF['parsed_age'].map(_map_age)
	else:
		raise ValueError("unknown dataset")

	# ensure reproducibility of results
	_RANDOM_SEED = 42
	np.random.seed(_RANDOM_SEED)
	print(f"Numpy seed set to {_RANDOM_SEED} for reproducibility.")

	_eval_downstream_task(_INDEX_DF, dataset, _TASK)
