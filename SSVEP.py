import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

import mne
import pdb
import scipy
# Load raw data
#I loaded data from the directory.
#data_path = mne.datasets.ssvep.data_path()
#bids_fname = (
    #data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
#)
bids_fname = ("record-[Subject-TestData1-Run3-2024.06.04-16.57.55].gdf")
raw = mne.io.read_raw_gdf(bids_fname, preload=True, verbose=False)
raw.info["line_freq"] = 50.0

# Set montage
montage = mne.channels.make_standard_montage("easycap-M1")
rename_dict = {
    "FP1":"Fp1",
    "FP2":"Fp2"
    }
#Rename channels (for mne naming system)
raw.rename_channels(rename_dict)
montage_path = 'C:/Users/ckrogmei/Documents/My Important Documents/FlickerFudge/Real_Test_Data/CACS-32_NO_REF.bvef'
#montage = mne.channels.read_custom_montage(montage_path)

montage = mne.channels.make_standard_montage("standard_1020")

raw.set_montage(montage, verbose=False)


#set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Apply bandpass filter
raw.notch_filter(np.arange(50,250,50),filter_length='auto',phase='zero')    #European notch
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)


#"33024": "VidA1", "33025": "VidA2"
#                        , "33026": "VidA3", "33027": "VidA4"
# "33048": "VidC3Intro"

# Construct epochs
raw.annotations.rename({ "33028": "VidB1", "33029": "VidB2"
                        , "33030": "VidC1", "33031": "VidC2"
                        , "33032": "VidD1", "33033": "VidD3"
                        , "33034": "VidD4", "33035": "VidE1"
                        , "33036": "VidE2", "33037": "VidE3"
                        , "33038": "VidF1", "33039": "VidF2"
                        , "33040": "VidG1", "33041": "VidG2"
                        , "33042": "VidJ1", "33043": "VidJ2"
                        , "33044": "VidH1", "33045": "VidH2"
                        , "33046": "VidH3", "33047": "VidH4"
                        , "32769": "ExpStart"})

                       # , "32775": "BaseStart"
                       # , "800": "TrialEnd", "32770": "ExpStop"})
tmin, tmax = -5.0, 10.0  # in s, changed to 10.0 from 20.0
baseline = None     #add baseline
epochs = mne.Epochs(
    raw,
    event_id=["VidB1", "VidB2",
              "VidC1", "VidC2", "VidD1", "VidD3", "VidD4", "VidE1",
              "VidE2", "VidE3", "VidF1", "VidF2", "VidG1", "VidG2",
              "VidJ1", "VidJ2", "VidH1", "VidH2", "VidH3", "VidH4",
              "ExpStart"], #"BaseStart", "TrialEnd", "ExpStop"],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    #event_repeated="keep",   #events repeated, and multiple events at same timestamp
    verbose=False,
)

# Extract and print events
events = epochs.events
event_dict = epochs.event_id

print("Event IDs and their corresponding events:")
for event_code, event_name in event_dict.items():
    print(f"{event_name}: {event_code}")

print("\nEvents from the epochs object:")
for event in events:
    print(event)

#exit()

#Compute power spectral density (PSD) and signal to noise ratio (SNR)

tmin = 1.0 #cut for transient stimulus onset response
tmax = 10.0 #changed to 10.0 from 20.0
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]

spectrum = epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)

def splitBaselinesStimuli(epochs,electrodeNum = 0, vidNum=0,fs=500,basetime=5):
    #This function gets baseline and evoked data. Notes, time shoud be set to -5 to begin. -5 to 0 is the baseline interval (non-inclusive 0).
    #vidNum is video stimulus to grab, as given in event_dict
    #electrodeNum is the number corresponding to the electrode you want to get.
    trialData = epochs.get_data()                #This creates a (n_videoxn_electrodex_trialTime) matrix
    nSamp = int(np.round(fs*basetime))        #Number of samples to 0.
    data2split = np.squeeze(trialData[vidNum,electrodeNum,:])    #Get the video,electrode combo. squeeze is to collapse into a 1-D array of eeg samples
    baseline = data2split[0:nSamp]               #Since the value nSamp is truncated, nSamp last value is not included (not-inclusive). For example, if nsamp = 400, data would be indices 0-399 (400 total samples)
    stimulus = data2split[nSamp:len(data2split)]
    return baseline,stimulus

def welchSpectro(timeSeries,fs=500,pltflag=1):
    f,pxx = scipy.signal.welch(timeSeries,fs)
    if pltflag:
        plt.plot(f,pxx)
        plt.show()
    return f,pxx
#--------------------------------------------------------------------------------------------------------------------------
#Let's split data here!
#--------------------------------------------------------------------------------------------------------------------------
baseline1,stimulus1 = splitBaselinesStimuli(epochs,electrodeNum=1,vidNum=2)
baseline2,stimulus2 = splitBaselinesStimuli(epochs,electrodeNum=1,vidNum=3)
f1,pxx1 = welchSpectro(baseline1,fs=500,pltflag=0)
f2,pxx2 = welchSpectro(baseline2,fs=500,pltflag=0)
f3,pxx3 = welchSpectro(stimulus1,fs=500,pltflag=0)
f4,pxx4 = welchSpectro(stimulus2,fs=500,pltflag=0)
fig, axs = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
axs[0,0].plot(f1,pxx1)
axs[0,0].set_title("Frequency-Baseline Video H1 electorde 1")
axs[0,1].plot(f3,pxx3)
axs[0,1].set_title("Frequency-Baseline Video H3")
axs[1,0].plot(f2,pxx2)
axs[1,0].set_title("Frequency-Stimulus Video H1 (Whoa, thats a nice 6Hz event")
axs[1,1].plot(f4,pxx4)
axs[1,0].set_title("Frequency-Stimulus Video H3 (Whoa, thats a nice but less nice 6Hz event")
plt.show()
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

#call function to compute SNR spectrum
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

#print(snrs[0].shape)


#trials, channels, freq bins
print(snrs.shape)

#exit()




#Above, we want to compare power at each bin with average
#    power of the three neighboring bins (on each side) and skip one bin directly next to it.




#Plotting SNR & PSD Spectra

fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    ylim=[-2, 30],
    xlim=[fmin, fmax],
)

fig.show()

pdb.set_trace()
# define stimulation frequency
stim_freq = 12.0

# find index of frequency bin closest to stimulation frequency
i_bin_12hz = np.argmin(abs(freqs - stim_freq))
# could be updated to support multiple frequencies

# for later, we will already find the 15 Hz bin and the 1st and 2nd harmonic
# for both.
i_bin_24hz = np.argmin(abs(freqs - 24))
i_bin_36hz = np.argmin(abs(freqs - 36))
i_bin_15hz = np.argmin(abs(freqs - 15))
i_bin_30hz = np.argmin(abs(freqs - 30))
i_bin_45hz = np.argmin(abs(freqs - 45))

i_trial_12hz = np.where(epochs.annotations.description == "VidD1")[0]
i_trial_15hz = np.where(epochs.annotations.description == "VidH3")[0]

# Define different ROIs
roi_vis = [
    "POz",
    "Oz",
    "O1",
    "O2",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "PO9",
    "PO10",
    "O9",
    "O10",
]  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(
    epochs.info, eeg=True, stim=False, exclude="bads", selection=roi_vis
)

snrs_target = snrs[i_trial_12hz, :, i_bin_12hz][:, picks_roi_vis]
print("sub 2, VidD1 trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target.mean()}")

# get average SNR at 12 Hz for ALL channels
snrs_12hz = snrs[i_trial_12hz, :, i_bin_12hz]
snrs_12hz_chaverage = snrs_12hz.mean(axis=0)

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_12hz_chaverage, epochs.info, vlim=(1, None), axes=ax)

print("sub 2, VidD1 trials, SNR at 12 Hz")
print(f"average SNR (all channels): {snrs_12hz_chaverage.mean()}")
print(f"average SNR (occipital ROI): {snrs_target.mean()}")

tstat_roi_vs_scalp = ttest_rel(snrs_target.mean(axis=1), snrs_12hz.mean(axis=1))
print(
    "12 Hz SNR in occipital ROI with VidD1 is significantly larger than 12 Hz SNR over all "
    f"channels: t = {tstat_roi_vs_scalp[0]:.3f}, p = {tstat_roi_vs_scalp[1]}"
)

#next is statistical separation of 2 things
#should pick 6 and 12 hz videos




#Here, doing below

snrs_roi = snrs[:, picks_roi_vis, :].mean(axis=1)

freq_plot = [12, 15, 24, 30, 36, 45]
color_plot = ["darkblue", "darkgreen", "mediumblue", "green", "blue", "seagreen"]
xpos_plot = [-5.0 / 12, -3.0 / 12, -1.0 / 12, 1.0 / 12, 3.0 / 12, 5.0 / 12]
fig, ax = plt.subplots()
labels = ["12 Hz trials", "15 Hz trials"]
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
res = dict()

# loop to plot SNRs at stimulation frequencies and harmonics
for i, f in enumerate(freq_plot):
    # extract snrs
    stim_12hz_tmp = snrs_roi[i_trial_12hz, np.argmin(abs(freqs - f))]
    stim_15hz_tmp = snrs_roi[i_trial_15hz, np.argmin(abs(freqs - f))]
    SNR_tmp = [stim_12hz_tmp.mean(), stim_15hz_tmp.mean()]
    # plot (with std)
    ax.bar(
        x + width * xpos_plot[i],
        SNR_tmp,
        width / len(freq_plot),
        yerr=np.std(SNR_tmp),
        label="%i Hz SNR" % f,
        color=color_plot[i],
    )
    # store results for statistical comparison
    res["stim_12hz_snrs_%ihz" % f] = stim_12hz_tmp
    res["stim_15hz_snrs_%ihz" % f] = stim_15hz_tmp

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("SNR")
ax.set_title("Average SNR at target frequencies")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(["%i Hz" % f for f in freq_plot], title="SNR at:")
ax.set_ylim([0, 70])
ax.axhline(1, ls="--", c="r")
fig.show()

