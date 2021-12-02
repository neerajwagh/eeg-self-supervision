import numpy as np
import torch
from torch import nn
from torch.nn import init

def square(x):
	return x * x


def safe_log(x, eps=1e-6):
	""" Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
	return torch.log(torch.clamp(x, min=eps))


def identity(x):
	return x


def squeeze_final_output(x):

	"""Removes empty dimension at end and potentially removes empty time
	 dimension. It does  not just use squeeze as we never want to remove
	 first dimension.
	Returns
	-------
	x: torch.Tensor
		squeezed tensor
	"""
	# print(x.size())
	assert x.size()[3] == 1
	x = x[:, :, :, 0]
	if x.size()[2] == 1:
		x = x[:, :, 0]
	return x

def transpose_time_to_spat(x):
	"""Swap time and spatial dimensions.
	Returns
	-------
	x: torch.Tensor
		tensor in which last and first dimensions are swapped
	"""
	return x.permute(0, 3, 2, 1)

def np_to_th(
	X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
	"""
	Convenience function to transform numpy array to `torch.Tensor`.
	Converts `X` to ndarray using asarray if necessary.
	Parameters
	----------
	X: ndarray or list or number
		Input arrays
	requires_grad: bool
		passed on to Variable constructor
	dtype: numpy dtype, optional
	var_kwargs:
		passed on to Variable constructor
	Returns
	-------
	var: `torch.Tensor`
	"""
	if not hasattr(X, "__len__"):
		X = [X]
	X = np.asarray(X)
	if dtype is not None:
		X = X.astype(dtype)
	X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
	if pin_memory:
		X_tensor = X_tensor.pin_memory()
	return X_tensor


class Ensure4d(torch.nn.Module):
	def forward(self, x):
		while(len(x.shape) < 4):
			x = x.unsqueeze(-1)
		return x


class Expression(torch.nn.Module):
	"""Compute given expression on forward pass.
	Parameters
	----------
	expression_fn : callable
		Should accept variable number of objects of type
		`torch.autograd.Variable` to compute its output.
	"""

	def __init__(self, expression_fn):
		super(Expression, self).__init__()
		self.expression_fn = expression_fn

	def forward(self, *x):
		return self.expression_fn(*x)

	def __repr__(self):
		if hasattr(self.expression_fn, "func") and hasattr(
			self.expression_fn, "kwargs"
		):
			expression_str = "{:s} {:s}".format(
				self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
			)
		elif hasattr(self.expression_fn, "__name__"):
			expression_str = self.expression_fn.__name__
		else:
			expression_str = repr(self.expression_fn)
		return (
			self.__class__.__name__ +
			"(expression=%s) " % expression_str
		)


class ShallowFBCSPNet(nn.Sequential):
	"""Shallow ConvNet model from Schirrmeister et al 2017.
	Model described in [Schirrmeister2017]_.
	Parameters
	----------
	in_chans : int
		XXX
	References
	----------
	.. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
	   L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
	   & Ball, T. (2017).
	   Deep learning with convolutional neural networks for EEG decoding and
	   visualization.
	   Human Brain Mapping , Aug. 2017.
	   Online: http://dx.doi.org/10.1002/hbm.23730
	"""

	def __init__(
		self,
		in_chans,
		n_classes,
		input_window_samples=None,
		n_filters_time=40,
		filter_time_length=25,
		n_filters_spat=40,
		pool_time_length=75,
		pool_time_stride=15,
		final_conv_length=30,
		conv_nonlin=square,
		pool_mode="mean",
		pool_nonlin=safe_log,
		split_first_layer=True,
		batch_norm=True,
		batch_norm_alpha=0.1,
		drop_prob=0.5,
	):
		super().__init__()
		if final_conv_length == "auto":
			assert input_window_samples is not None
		self.in_chans = in_chans
		self.n_classes = n_classes
		self.input_window_samples = input_window_samples
		self.n_filters_time = n_filters_time
		self.filter_time_length = filter_time_length
		self.n_filters_spat = n_filters_spat
		self.pool_time_length = pool_time_length
		self.pool_time_stride = pool_time_stride
		self.final_conv_length = final_conv_length
		self.conv_nonlin = conv_nonlin
		self.pool_mode = pool_mode
		self.pool_nonlin = pool_nonlin
		self.split_first_layer = split_first_layer
		self.batch_norm = batch_norm
		self.batch_norm_alpha = batch_norm_alpha
		self.drop_prob = drop_prob

		self.add_module("ensuredims", Ensure4d())
		pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
		if self.split_first_layer:
			self.add_module("dimshuffle", Expression(transpose_time_to_spat))
			self.add_module(
				"conv_time",
				nn.Conv2d(
					1,
					self.n_filters_time,
					(self.filter_time_length, 1),
					stride=1,
				),
			)
			self.add_module(
				"conv_spat",
				nn.Conv2d(
					self.n_filters_time,
					self.n_filters_spat,
					(1, self.in_chans),
					stride=1,
					bias=not self.batch_norm,
				),
			)
			n_filters_conv = self.n_filters_spat
		else:
			self.add_module(
				"conv_time",
				nn.Conv2d(
					self.in_chans,
					self.n_filters_time,
					(self.filter_time_length, 1),
					stride=1,
					bias=not self.batch_norm,
				),
			)
			n_filters_conv = self.n_filters_time
		if self.batch_norm:
			self.add_module(
				"bnorm",
				nn.BatchNorm2d(
					n_filters_conv, momentum=self.batch_norm_alpha, affine=True
				),
			)
		self.add_module("conv_nonlin_exp", Expression(self.conv_nonlin))
		self.add_module(
			"pool",
			pool_class(
				kernel_size=(self.pool_time_length, 1),
				stride=(self.pool_time_stride, 1),
			),
		)
		self.add_module("pool_nonlin_exp", Expression(self.pool_nonlin))
		self.add_module("drop", nn.Dropout(p=self.drop_prob))
		self.eval()
		if self.final_conv_length == "auto":
			out = self(
				np_to_th(
					np.ones(
						(1, self.in_chans, self.input_window_samples, 1),
						dtype=np.float32,
					)
				)
			)
			n_out_time = out.cpu().data.numpy().shape[2]
			self.final_conv_length = n_out_time
		self.add_module(
			"conv_classifier",
			nn.Conv2d(
				n_filters_conv,
				self.n_classes,
				(self.final_conv_length, 1),
				bias=True,
			),
		)

		self.add_module("squeeze", Expression(squeeze_final_output))

		# Initialization, xavier is same as in paper...
		init.xavier_uniform_(self.conv_time.weight, gain=1)
		# maybe no bias in case of no split layer and batch norm
		if self.split_first_layer or (not self.batch_norm):
			init.constant_(self.conv_time.bias, 0)
		if self.split_first_layer:
			init.xavier_uniform_(self.conv_spat.weight, gain=1)
			if not self.batch_norm:
				init.constant_(self.conv_spat.bias, 0)
		if self.batch_norm:
			init.constant_(self.bnorm.weight, 1)
			init.constant_(self.bnorm.bias, 0)
		init.xavier_uniform_(self.conv_classifier.weight, gain=1)
		init.constant_(self.conv_classifier.bias, 0)



class RelativePositioningShallowNet(nn.Module):
	"""Contrastive module with linear layer on top of siamese embedder.

	Parameters
	----------
	emb : nn.Module
		Embedder architecture.
	emb_size : int
		Output size of the embedder.
	dropout : float
		Dropout rate applied to the linear layer of the contrastive module.
	"""
	def __init__(self, args):
		super().__init__()
		self.embedder = ShallowFBCSPNet(

				in_chans=19,

				n_classes=100,

				input_window_samples=800,

				n_filters_time=40,
				filter_time_length=25,
				n_filters_spat=40,
				pool_time_length=75,
				pool_time_stride=15,
				# final_conv_length=30,
				final_conv_length="auto",
				conv_nonlin=square,
				pool_mode="mean",
				pool_nonlin=safe_log,
				split_first_layer=True,
				batch_norm=True,
				batch_norm_alpha=0.1,
				drop_prob=0.5,
		)

		self.clf = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(100, 1)
		)

		self.loss = torch.nn.SoftMarginLoss()

	def forward(self, x1, x2, y_pseudo):
		z1, z2 = self.embedder(x1), self.embedder(x2)
		y_predicted = self.clf(torch.abs(z1 - z2)).flatten()
		pretext_loss = self.loss(torch.squeeze(y_predicted), y_pseudo)
		return pretext_loss

	def forward_outputs(self, x):
		z = self.embedder(x)
        # CAUTION: this classification layer is added in the training code
        # NOTE: self.clf is ignored
		outputs = self.fc(z)
		return outputs
