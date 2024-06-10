import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

import mne
import pdb
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
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)


