from collections import OrderedDict
import numpy as np 

_BRAIN_RHYTHMS = OrderedDict({
	"delta" : [1.0, 4.0],
	"theta": [4.0, 8.0],
	"alpha": [8.0, 13.0],
	"beta": [13.0, 25.0],
	"gamma": [25.0, 40.0]
})

def eeg_power_in_bands(epochs, relative=True, freq_bands=_BRAIN_RHYTHMS):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.
    relative : Whether returned values are relative
    freq_bands: frequency bands to calculate PSD features

    Returns
    -------
    tot_power : total power per channel
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """


    epo_spectrum = epochs.compute_psd(fmin=1.0, fmax=45., method='welch')
    psds, freqs = epo_spectrum.get_data(return_freqs=True)

    tot_power = np.sum(psds, axis=-1, keepdims=True)
    if relative:
        # Normalize the PSDs
        psds /= tot_power

    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].sum(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    tot_power = np.squeeze(tot_power, axis=-1)
    X = np.array(X)
    
    return tot_power, X