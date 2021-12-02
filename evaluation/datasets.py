import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, window_idx, X, y):
        self.window_idx = window_idx
        self.topo_data = X
        self.labels = y
    
    def __len__(self):
        return len(self.window_idx)
    
    def __getitem__(self, sample_idx):
        window_idx = self.window_idx[sample_idx]
        return {
				"window_idx": window_idx,
				"feature_data": torch.from_numpy(self.topo_data[window_idx, ...]),
                "label": self.labels[window_idx]
		}


class TimeseriesDataset(torch.utils.data.Dataset):
	def __init__(self, window_idx, X, y):
		self.window_idx = window_idx
		self.timeseries_data = X
		self.labels = y
	
	def __len__(self):
		return len(self.window_idx)
	
	def __getitem__(self, sample_idx):
		window_idx = self.window_idx[sample_idx]
		return {
			"window_idx": window_idx,
			"feature_data": torch.from_numpy(self.timeseries_data[window_idx, ...]),
			"label": self.labels[window_idx]
		}
