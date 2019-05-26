import numpy as np
from scipy.signal import (butter, filtfilt, cheby1, iirfilter, lfilter)
import scipy
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

from six import string_types, integer_types

_ACCEPTED_ARRAY_DTYPES = (np.float, np.float32, np.float64,
                          np.int, np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          np.bool)



# Required input defintions are as follows;
# time:   Time between samples
# band:   The bandwidth around the centerline freqency that you wish to filter
# freq:   The centerline frequency to be filtered
# ripple: The maximum passband ripple that is allowed in db
# order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
#         IIR filters are best suited for high values of order.  This algorithm
#         is hard coded to FIR filters
# filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
# data:         the data to be filtered
def Implement_Notch_Filter(time, band, freq, data, order=4):
    fs   = 1000
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], btype='bandstop',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    y = filtfilt(b, a, np.float32(data))
    return y


# align data to mean
def align_get_shifts_with_ref(wf, ref, upsample_factor = 5, nshifts = 115):

    ''' Align all waveforms on a single channel    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)
    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = upsample_resample(ref[np.newaxis], upsample_factor)[0]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts/np.float32(upsample_factor)

def upsample_resample(wf, upsample_factor):
    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = scipy.signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces


def shift_chans(wf, best_shifts):
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    # Cat: TODO read this from CNOFIG
    wf_shifted = []
    wfs_final=[]
    for k, shift_ in enumerate(best_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final.append(temp)
    wf_shifted = np.array(wfs_final)
    
    return wf_shifted



class crop(object):
    def __init__(self,data, fname):
        self.data = data
        self.fname = fname

        self.define_ROI()

    def define_ROI(self):

        #if os.path.exists(self.fname[:-4]+".npy")==False:
        roi_coords = self.select_ROI()

        return (roi_coords) 

    def select_ROI(self):
        ''' Function to crop field-of-view of video
        '''

        self.fig, self.ax = plt.subplots()
        self.coords=[]

        self.sample_image = self.data[1]
        print (self.sample_image.shape)
        self.ax.imshow(self.sample_image)#, vmin=0.0, vmax=0.02)
        #self.ax.set_title(ROI_name)

        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show(block=False)

    def on_click(self, event):
        ''' Mouse click function that catches clicks and plots them on top of existing image
        '''

        if event.inaxes is not None:
            print (event.ydata, event.xdata)
            self.coords.append((event.ydata, event.xdata))
            #for j in range(len(self.coords)):
            for k in range(2):
                for l in range(2):
                    self.sample_image[int(event.ydata)-1+k,int(event.xdata)-1+l]=np.max(self.sample_image)

            self.ax.imshow(self.sample_image)
            self.fig.canvas.draw()
        else:
            print ('Exiting')
            plt.close()
            #self.fig.canvas.mpl_disconnect(self.cid)
            print (" saving coords: ", self.fname[:-4]+"_coords.npy")
            np.save(self.fname[:-4]+"_coords.npy", self.coords)

            return

# load coords and visualized cropped data
#f_out = fname[:-4]+"_brainmask.npy"
#if os.path.exists(f_out)==False:
def make_mask(data, fname):
    #data = np.load(fname).transpose(2,0,1)
    coords = np.load(os.path.split(fname)[0]+"/coords.npy")
    
    #Search points outside mask
    all_points = []
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            all_points.append([i,j])

    all_points = np.array(all_points)
    vertixes = np.array(coords) 
    vertixes_path = Path(vertixes)

    mask = vertixes_path.contains_points(all_points)
    print (mask.shape)
    counter=0
    coords_save=[]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if mask[counter] == False:
                #images_processed[i][j]=np.nan
                data[:,i,j]=0
                coords_save.append([i,j])
            counter+=1

    #final_mask = images_processed
    #np.save(fname[:-4]+"_brainmask.npy", final_mask)
    #fig, ax = plt.subplots()
    #ax.imshow(images_processed)
    #plt.show()
    
    return data
  
  # Kwikteam / phy CCG computation module
# Adapted by Catalin Mitelut from github code last updated March 23, 2016

# -*- coding: utf-8 -*-

"""Cross-correlograms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.
    Implicitely assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]
    
def _unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]

def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.
    Avoid a copy if possible.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out


def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _as_array(arr)
    indices = _as_array(indices)
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = _as_array(arr)
    return arr[steps:] - arr[:len(arr) - steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1),
                    dtype=np.int32)


def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                      correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def correlograms(spike_times,
                 spike_clusters,
                 cluster_ids=None,
                 sample_rate=1.,
                 bin_size=None,
                 window_size=None,
                 symmetrize=True,
                 ):
    """Compute all pairwise cross-correlograms among the clusters appearing
    in `spike_clusters`.
    Parameters
    ----------
    spike_times : array-like
        Spike times in seconds.
    spike_clusters : array-like
        Spike-cluster mapping.
    cluster_ids : array-like
        The list of unique clusters, in any order. That order will be used
        in the output array.
    bin_size : float
        Size of the bin, in seconds.
    window_size : float
        Size of the window, in seconds.
    Returns
    -------
    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise
        CCGs.
    """
    assert sample_rate > 0.
    assert np.all(np.diff(spike_times) >= 0), ("The spike times must be "
                                               "increasing.")

    # Get the spike samples.
    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_samples = (spike_times * sample_rate).astype(np.int64)

    
    spike_clusters = _as_array(spike_clusters)

    assert spike_samples.ndim == 1
    assert spike_samples.shape == spike_clusters.shape

    # Find `binsize`.
    bin_size = np.clip(bin_size, 1e-5, 1e5)  # in seconds
    binsize = int(sample_rate * bin_size)  # in samples
    
    assert binsize >= 1

    # Find `winsize_bins`.
    window_size = np.clip(window_size, 1e-5, 1e5)  # in seconds
    winsize_bins = 2 * int(.5 * window_size / bin_size) + 1

    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    # Take the cluster oder into account.
    if cluster_ids is None:
        clusters = _unique(spike_clusters)
    else:
        clusters = _as_array(cluster_ids)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=np.bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # # Update the masks given the clusters to update.
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index((spike_clusters_i[:-shift][m],
                                        spike_clusters_i[+shift:][m],
                                        d),
                                       correlograms.shape)

        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        shift += 1

    # Remove ACG peaks.
    correlograms[np.arange(n_clusters),
                 np.arange(n_clusters),
                 0] = 0

    if symmetrize:
        return _symmetrize_correlograms(correlograms)
    else:
        return correlograms


def compute_correlogram(units, spike_train, sample_rate=20000, bin_width = 0.005, window_size = 0.5):

    #Reduce spike_train to two units; ensure to keep order of spikes 
    #spike_train_temp = []
    #for unit in units:
    #    print (unit)
    #spike_train_temp.append(spike_train[np.where(spike_train[:,1]==unit)[0]])
    
    #spike_train = np.vstack(spike_train_temp)
    order_indexes = np.argsort(spike_train[:,0])
    spike_train = spike_train[order_indexes]
    
    return correlograms(spike_train[:,0]/float(sample_rate),spike_train[:,1],sample_rate=sample_rate, bin_size=bin_width, window_size=window_size)

