#%% Import Libraries ##########################################
import mne
import numpy as np
from mne.decoding import GeneralizingEstimator, SlidingEstimator, cross_val_multiscore, get_coef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import gc
from scipy.fftpack import fft, fftfreq
from mne.filter import resample
from Utility_Functions_V2 import resample_epochs_match_target
import seaborn as sns
from mne.decoding import CSP
from sklearn.model_selection import StratifiedKFold

# %% Parameters

# Developer Mode
Developer_Mode = False
New_Sample_Freq = 25

# Initialisation:
MagOrGrad = 'grad'
Filter_LowLimit = 0.1
Filter_HighLimit = 40

#Cropping
Crop_min_Training = 0 #seconds
Crop_max_Training = None #seconds

# Participants
participant_ids = ["150"]

#Baseline
Apply_Baseline = True
Baseline_Window_Training = (-0.1, 0) #seconds range

# Graph Output Directories
plot_dir = "home/jm283064/"
decoding_dir = plot_dir + "DecodingAccuracy"
generalisation_dir = plot_dir + "GeneralisationMatrix"
testauc_dir = plot_dir + "TestAUC"
for d in [decoding_dir, generalisation_dir, testauc_dir]:
    os.makedirs(d, exist_ok=True)

# %% MAIN FUNCTION

def RUN_PARTICIPANT(long_data_file, short_data_file, MagOrGrad, Filter_LowLimit, Filter_HighLimit,
                    Apply_Baseline, Baseline_Window,
                    Crop_min_Training, Crop_max_Training,
                    Developer_Mode, New_Sample_Freq,
                    resample_epochs_match_target ):
    
    return 