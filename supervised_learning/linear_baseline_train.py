import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils import _custom_cv_fold_iterator, _get_subject_level_split, _map_age


def _train_downstream_task(index_df, task, dataset, seed):

	# load X based on dataset
	X = np.load(f'psd_bandpower_relative_{dataset}.npy').astype(np.float32)
	X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

	#load y based on given dataset and task
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

	# map labels to 0/1
	label_mapping, y = np.unique(y, return_inverse=True)
	print(f"Unique labels 0/1 mapping: {label_mapping}")

	# make subject level split
	all_subjects, train_subjects, val_subjects, heldout_subjects, train_window_idx, val_window_idx, heldout_window_idx = \
	_get_subject_level_split(index_df, _RANDOM_SEED, dataset, task)
	
	# linear baseline will use train and validation subjects for training. Other pipelines only use train subjects.
	train_and_val_window_idx = np.concatenate((train_window_idx, val_window_idx), axis=0)
	train_and_val_subjects = np.concatenate((train_subjects, val_subjects), axis=0)
	
	print("train_subjects: ", len(train_subjects), " val subjects: ", len(val_subjects), \
		" heldout_subjects: ", len(heldout_subjects), " all subjects: ", len(all_subjects))
	print(f"len train_window_idx: {len(train_window_idx)}, len val_window_idx: {len(val_window_idx)}, len heldout_window_idx:{len(heldout_window_idx)}")

	# Load train and validation windows
	X_train_and_val = X[train_and_val_window_idx, ...]
	y_train_and_val = y[train_and_val_window_idx, ...]

	# initialize logistic regression pipeline
	pipeline = Pipeline([
		('clf', LogisticRegression(
			solver='saga',
			penalty='elasticnet', 
			class_weight='balanced',
			max_iter=500,
			n_jobs=-1,
			random_state=_RANDOM_SEED
			)
		)
	])

	# define parameters for grid search
	param_grid = {
		# clf_C for logisticRegression
		'clf__C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
		'clf__l1_ratio': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
	}

	# initialize grid search
	grid = GridSearchCV(
		pipeline, 
		cv=_custom_cv_fold_iterator(train_and_val_subjects=train_and_val_subjects, num_folds=5, task=task, index_df=index_df, dataset=dataset, _RANDOM_SEED=seed), 
		scoring='roc_auc',
		param_grid=param_grid, 
		n_jobs=12,
		refit=False, # CAUTION: since we're passing full X,y to fit(), it will retrain on heldout window idx too (even though CV is correct)! DO NOT KEEP refit='AUC'!
		verbose=3
	)

	grid.fit(X, y)

	# summarize results
	print(f"Best: {grid.best_score_} using {grid.best_params_}")
	best_params = {x.replace("clf__", ""): v for x, v in grid.best_params_.items()}

	# define the best fit model
	final_model = LogisticRegression(
					solver='saga',
					penalty='elasticnet', 
					class_weight='balanced',
					max_iter=500,
					n_jobs=-1,
					random_state=_RANDOM_SEED,
					**best_params
	).fit(X_train_and_val, y_train_and_val)

	# save the best model in a pkl file
	joblib.dump(final_model, f'{dataset}_{task}_linear.pkl')
	return

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Execute evaluation pipeline using baseline PSD features.")
	parser.add_argument('--dataset', type=str, help="choose between tuh and lemon")
	parser.add_argument('--task', type=str, help="choose from ...")
	args = parser.parse_args()

	dataset = args.dataset
	task= args.task

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

	_train_downstream_task(_INDEX_DF, task, dataset, _RANDOM_SEED)

	print("done")
