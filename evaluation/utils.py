import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _map_age(x):
	if x == -1:
		return "n/a"
	elif x <= 45:
		return "young"
	else:
		return "old"

def _get_subject_level_split(index_df, random_seed, dataset, _TASK):

	if dataset == "tuh":
		if _TASK == "gender":
			task_name = 'parsed_gender'
		elif _TASK == "age":
			task_name = 'parsed_age_binary'
		elif _TASK == "condition":
			task_name = "text_label"
		else:
			raise ValueError('unknown task!')
		
		all_subjects = index_df["patient_id"].unique().tolist()

		y = [index_df.loc[index_df["patient_id"] == x][task_name].values[0] for x in all_subjects]

		rest_subjects, heldout_subjects = train_test_split(all_subjects, test_size=0.3, shuffle=True, random_state=random_seed, stratify=y)

		y_rest = [index_df.loc[index_df["patient_id"] == x][task_name].values[0] for x in rest_subjects]
		train_subjects, val_subjects = train_test_split(rest_subjects, test_size=0.2, shuffle=True, random_state=random_seed, stratify=y_rest)

		train_window_idx = index_df[index_df["patient_id"].isin(train_subjects)]['new_window_idx'].to_numpy()
		val_window_idx = index_df[index_df["patient_id"].isin(val_subjects)]['new_window_idx'].to_numpy()
		heldout_window_idx = index_df[index_df["patient_id"].isin(heldout_subjects)]['new_window_idx'].to_numpy()
		
	elif dataset == "lemon":
		if _TASK == "gender":
			task_name = 'gender'
		elif _TASK == "age":
			task_name = 'age'
		elif _TASK == "condition":
			task_name = "condition"
		else:
			raise ValueError('unknown task!')
		all_subjects = index_df["subject_id"].unique().tolist()

		y = [index_df.loc[index_df['subject_id'] == x][task_name].values[0] for x in all_subjects]
		rest_subjects, heldout_subjects = train_test_split(all_subjects, test_size=0.3, shuffle=True, random_state=random_seed, stratify=y)

		y_rest = [index_df.loc[index_df['subject_id'] == x][task_name].values[0] for x in rest_subjects]

		train_subjects, val_subjects = train_test_split(rest_subjects, test_size=0.2, shuffle=True, random_state=random_seed, stratify=y_rest)

		train_window_idx = index_df[index_df['subject_id'].isin(train_subjects)]['window_idx'].to_numpy()
		val_window_idx = index_df[index_df['subject_id'].isin(val_subjects)]['window_idx'].to_numpy()
		heldout_window_idx = index_df[index_df['subject_id'].isin(heldout_subjects)]['window_idx'].to_numpy()
	
	else:
		raise ValueError("unknown dataset")

	return all_subjects, train_subjects, val_subjects, heldout_subjects, train_window_idx, val_window_idx, heldout_window_idx

# create patient-level metrics
def get_patient_prediction(df, dataset):

	if dataset == "lemon":
		title = "subject_id"
	elif dataset == "tuh":
		title = "patient_id"
	else:
		raise ValueError("unknown dataset")
	
	# NOTE: grouping based on raw file is better than grouping by patient_id since the same patient can have multiple shards, each with different labels
	# CAUTION: labels are only consistent inside a shard, not necessarily for all shards of the same subject
	grouped_df = df.groupby("raw_file_path")
	rows = [ ]
	for raw_file_path, shard_df in grouped_df:
		
		assert len(list(shard_df[title].unique())) == 1
		patient_id = shard_df[title].values[0]

		temp = { }
		temp[title] = patient_id
		temp["y_true"] = list(shard_df["y_true"].unique())[0]

		try:
			# NOTE: this assert will trigger with grouped_df = df.groupby("patient_id")
			# NOTE: some patients have both labels - multiple shards belonging to different labels - normal, abnormal
			# NOTE: for full list of all patient_ids that trigger this, see .txt file in same folder
			assert len(list(shard_df["y_true"].unique())) == 1
		except:
			print(f"SKIPPING PATIENT_ID FOR PATIENT-LEVEL METRICS: {patient_id}")
			continue

		temp["y_pred"] = shard_df["y_pred"].mode()[0]
		temp["y_probs_0"] = shard_df["y_probs_0"].mean()
		temp["y_probs_1"] = shard_df["y_probs_1"].mean()
		rows.append(temp)

	return_df = pd.DataFrame(rows)
	return np.array(list(zip(return_df["y_probs_0"], return_df["y_probs_1"]))), list(return_df["y_true"]), list(return_df["y_pred"])


def _custom_cv_fold_iterator(train_and_val_subjects, num_folds, _TASK, _INDEX_DF, _RANDOM_SEED):

	print(len(train_and_val_subjects))

	y = [_INDEX_DF.loc[_INDEX_DF['subject_id'] == x][_TASK].values[0] for x in train_and_val_subjects]
	print(len(y))

	cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=_RANDOM_SEED, )
	cv_iter = cv.split(range(len(train_and_val_subjects)), y)
	
	for (train_subject_idx, val_subject_idx), fold_idx in zip(cv_iter, range(num_folds)):
	
		train_subjects = train_and_val_subjects[train_subject_idx] # these indices are generated internally in sklearn KFold
		val_subjects = train_and_val_subjects[val_subject_idx]  # these indices are generated internally in sklearn KFold

		train_idx = _INDEX_DF.index[_INDEX_DF["subject_id"].astype("str").isin(train_subjects)].tolist()
		val_idx = _INDEX_DF.index[_INDEX_DF["subject_id"].astype("str").isin(val_subjects)].tolist()

		print(f"Fold {fold_idx}: train idx: {len(train_idx)} val idx: {len(val_idx)}")
		yield train_idx, val_idx

