#%% Import Libraries ##########################################
import os
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
Crop_max_Training = 1.6 #seconds

# Participants
participant_ids = ["150"]

#Baseline
Apply_Baseline = True
Baseline_Window_Training = (-0.1, 0) #seconds range

# Graph Output Directories
plot_dir = "home/jm283064/"
crossval_decoding_dir = plot_dir + "CrossValDecodingAccuracy"
generalisation_dir = plot_dir + "GeneralisationMatrix"
testauc_dir = plot_dir + "TestAUC"
for d in [crossval_decoding_dir, generalisation_dir, testauc_dir]:
    os.makedirs(d, exist_ok=True)

# %% SUBROUTINES

def Plot_AUC(generalization_matrix, test_times, save_path):

    mean_auc = np.mean(generalization_matrix, axis=0)
    std_auc = np.std(generalization_matrix, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(test_times, mean_auc, label="Mean AUC", color="r")
    plt.fill_between(test_times, mean_auc - std_auc, mean_auc + std_auc, 
                     color="r", alpha=0.2, label= "±1 SD" )
    plt.axhline(0.5, linestyle='--', color='gray', label="Chance Level")

    plt.xlabel("Test Time (s)")
    plt.ylabel("ROC AUC")
    plt.title("Decoding Accuracy Over Time")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.show()
    
    return(mean_auc, std_auc)

def Generalise_To_Test_Data(TrainEpochs, TestEpochs, Trained_Model, XTest, yTest, MagOrGrad, save_path):
    
    generalization_matrix = np.zeros((len(TrainEpochs.times), len(TestEpochs.times)))

    for i, model in enumerate(Trained_Model.estimators_):
        print(f"Evaluating model trained at {TrainEpochs.times[i]:.3f}s...")

        # Predict probabilities for every test time point
        y_pred_proba = np.array([
            model.predict_proba(XTest[:, :, t])[:, 1] for t in range(XTest.shape[2])
        ]).T  # Shape: (n_trials, n_timepoints)

        # Compute AUC for each test time point
        for j in range(len(TestEpochs.times)):
            generalization_matrix[i, j] = roc_auc_score(yTest, y_pred_proba[:, j])

    #  Plot Generalisation Heatmap

    Colour_Scheme = "coolwarm" #viridis" #"RdBlu_r", "coolwarm"

    # Adjust the time labels so the first time point is 0, ascending
    train_times = TrainEpochs.times - TrainEpochs.times[0]  
    test_times = TestEpochs.times - TestEpochs.times[0] 

    plt.figure(figsize=(12, 10))

    # Plot heatmap without explicit tick labels
    sns.heatmap(generalization_matrix, cmap=Colour_Scheme, center=0.5, cbar_kws={'label': 'Decoding AUC'}, vmin=0, vmax=1)

    # Set axis labels
    plt.ylabel("Training Time (s)", fontsize=14)
    plt.xlabel("Test Time (s)", fontsize=14)
    plt.title("Generalization Matrix: Model Training Time vs. Test Time: " + MagOrGrad.upper(), fontsize=16)

    # Manually select fewer tick positions for clarity
    num_ticks = 8
    y_tick_positions = np.linspace(0, len(train_times)-1, num_ticks, dtype=int)
    x_tick_positions = np.linspace(0, len(test_times)-1, num_ticks, dtype=int)

    # Set ticks and labels manually (add 0.5 for center alignment)
    plt.yticks(y_tick_positions + 0.5, labels=np.round(train_times[y_tick_positions], 2), rotation=0, fontsize=12)
    plt.xticks(x_tick_positions + 0.5, labels=np.round(test_times[x_tick_positions], 2), rotation=30, ha='right', fontsize=12)

    # Flip the y-axis to place 0 at the bottom
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    
    return(generalization_matrix)

def Train_Model(TrainEpochs, save_path):
    
    XTrain = TrainEpochs.get_data()  
    yTrain = TrainEpochs.events[:, -1]
    train_times = TrainEpochs.times  # Use original times for reference

    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    time_decod = SlidingEstimator(clf, scoring='roc_auc', n_jobs=-1)

    scores = cross_val_multiscore(time_decod, XTrain, yTrain, cv=5, n_jobs=-1)
    mean_crossval_scores = np.mean(scores, axis=0)

    # Plot decoding accuracy over time
    plt.figure(figsize=(8, 4))
    plt.plot(train_times, mean_crossval_scores, label="Cross Validation Decoding Accuracy CV=5", color="b")
    plt.axhline(0.5, linestyle='--', color='gray', label="Chance Level")
    plt.xlabel("Time (s)")
    plt.ylabel("Cross Validation Decoding Accuracy CV=5")
    plt.title("Decoding Accuracy per Timepoint: " + MagOrGrad.upper())
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)

    time_decod.fit(XTrain, yTrain)
    
    return(time_decod, mean_crossval_scores)

def Preprocess_Test_Epochs(Test_data_path, MagOrGrad, Apply_Baseline,
                           Developer_Mode, New_Sample_Freq):

    Test_epochs = mne.read_epochs(Test_data_path)
    Test_epochslong = Test_epochs["1_item/long"] 
    Test_epochsshort = Test_epochs["1_item/short"]

    if Apply_Baseline:
        Test_epochslong.apply_baseline((-4.2,-4.1))
        Test_epochsshort.apply_baseline((-2.2,-2.1))

    Test_epochslong.events[:, -1] = 1
    Test_epochsshort.events[:, -1] = 2

    Test_epochslong = mne.EpochsArray(Test_epochslong.get_data(), Test_epochslong.info, 
                                events=Test_epochslong.events, tmin=Test_epochslong.tmin, 
                                event_id={'long':1})

    Test_epochsshort = mne.EpochsArray(Test_epochsshort.get_data(), Test_epochsshort.info, 
                                events=Test_epochsshort.events, tmin=Test_epochsshort.tmin, 
                                event_id={'short':2})


    test_epochs_combined = mne.concatenate_epochs([Test_epochslong, Test_epochsshort])
    test_epochs_combined.pick(MagOrGrad)
    test_epochs_combined.filter(l_freq=0.1, h_freq=40, fir_design='firwin')
    TestEpochs = test_epochs_combined.copy().crop(tmin=0.5, tmax=4)

    if Developer_Mode:
        TestEpochs.resample(sfreq=New_Sample_Freq)

    XTest = TestEpochs.get_data()  
    yTest = TestEpochs.events[:, -1]
    
    return (XTest, yTest, Test_epochs)

def Preprocess_Train_Epochs (long_data_file, short_data_file, MagOrGrad,
                             Apply_Baseline, Baseline_Window, 
                             Developer_Mode,
                             Filter_LowLimit, Filter_HighLimit):

     # Read Epochs
    epochslong = mne.read_epochs(long_data_file)
    epochsshort = mne.read_epochs(short_data_file)

    # Pick Relevant Channels
    epochslong.pick(MagOrGrad)
    epochsshort.pick(MagOrGrad)

    # Filter Frequencies
    epochslong.filter(l_freq=Filter_LowLimit, h_freq=Filter_HighLimit, fir_design='firwin')
    epochsshort.filter(l_freq=Filter_LowLimit, h_freq=Filter_HighLimit, fir_design='firwin')

    # Apply Baseline
    if Apply_Baseline:
        epochslong.apply_baseline(Baseline_Window)
        epochsshort.apply_baseline(Baseline_Window)

    # Crop out Maintainance period
    epochslong.crop(epochslong.tmin, (epochslong.tmax-500))
    epochsshort.crop(epochsshort.tmin, (epochsshort.tmax-500))

    # Take the last 1.6s from each group of epochs
    epochslong.crop((epochslong.tmax-1.6), epochslong.tmax)
    epochsshort.crop(epochsshort.tmax-1.6, (epochsshort.tmax))

    # Set Event Labels (Trial Types)
    epochslong.events[:, -1] = 1
    epochsshort.events[:, -1] = 2
    epochslong = mne.EpochsArray(epochslong.get_data(), epochslong.info, 
                                 events=epochslong.events, tmin=epochslong.tmin, 
                                 event_id={'long':1})
    epochsshort = mne.EpochsArray(epochsshort.get_data(), epochsshort.info, 
                                  events=epochsshort.events, tmin=epochsshort.tmin, 
                                  event_id={'short':2})

    # Concantinate Long & Short Trials
    epochs_combined = mne.concatenate_epochs([epochslong, epochsshort])

    if Developer_Mode:
        epochs_combined.resample(sfreq=New_Sample_Freq)

    return epochs_combined


# %% MAIN FUNCTIONS

def RUN_PARTICIPANT(long_data_file, short_data_file, test_data_path, 
                    MagOrGrad, Filter_LowLimit, Filter_HighLimit,
                    Apply_Baseline, Baseline_Window,
                    Developer_Mode, New_Sample_Freq,
                    crossval_decoding_dir_PID, generalisation_dir_PID, testAUC_dir_PID):
    

   Train_Epochs = Preprocess_Train_Epochs(long_data_file, short_data_file, MagOrGrad,
                                          Apply_Baseline, Baseline_Window, 
                                          Developer_Mode, New_Sample_Freq,
                                          Filter_LowLimit, Filter_HighLimit)
   
   Trained_Model, mean_crosval_scores = Train_Model(Train_Epochs, crossval_decoding_dir_PID)
   
   XTest, yTest, TestEpochs = Preprocess_Test_Epochs(test_data_path, MagOrGrad, 
                          Apply_Baseline, Developer_Mode, New_Sample_Freq)
   
   generalisation_matrix = Generalise_To_Test_Data(Train_Epochs, TestEpochs, Trained_Model, 
                                                   XTest, yTest, MagOrGrad, generalisation_dir_PID)
   

   mean_test_auc, std_test_auc = Plot_AUC(generalisation_matrix, TestEpochs.times, testAUC_dir_PID)


   results = {
        'train_times': Train_Epochs.times,
        'test_times': TestEpochs.times,
        'mean_scores': mean_crosval_scores,
        'gen_matrix': generalisation_matrix,
        'mean_test_auc': mean_test_auc,
        'std_test_auc': std_test_auc
    }

   # Add saving to model generalisation

   
   return(results)

def Generate_Summary_Plots(all_participants_results, 
                           Crossval_OutputFolder, Generalisation_OutputFolder, AUC_OutputFolder):


    Traim_Time_Axis = all_participants_results[0]['train_times']
    Test_Time_Axis = all_participants_results[0]['test_times']

    average_decoding = np.mean([res['mean_scores'] for res in all_participants_results], axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(Traim_Time_Axis, average_decoding, color='b')
    plt.axhline(0.5, linestyle='--', color='gray')
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding Accuracy (AUC)")
    plt.title("Average Decoding Accuracy per Timepoint")
    plt.tight_layout()
    plt.savefig(os.path.join(Crossval_OutputFolder, 'average_decoding_accuracy.png'))

    average_gen_matrix = np.mean([res['gen_matrix'] for res in all_participants_results], axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(average_gen_matrix, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
    plt.xlabel("Test Time (s)")
    plt.ylabel("Train Time (s)")
    plt.title("Average Generalization Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(Generalisation_OutputFolder, 'average_generalization_matrix.png'))

    average_test_auc = np.mean([res['mean_test_auc'] for res in all_participants_results], axis=0)
    average_test_auc_std = np.mean([res['std_test_auc'] for res in all_participants_results], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(Test_Time_Axis, average_test_auc, color='r')
    plt.fill_between(Test_Time_Axis, average_test_auc - average_test_auc_std, average_test_auc + average_test_auc_std, color='r', alpha=0.2)
    plt.axhline(0.5, linestyle='--', color='gray')
    plt.xlabel("Time (s)")
    plt.ylabel("AUC Score")
    plt.title("Average Test AUC Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(AUC_OutputFolder, 'average_test_auc.png'))

    return()

# IMPLEMENTATION

all_participants_results = []

for id in participant_ids:

    Crossval_acc_plot_path = crossval_decoding_dir + "/" + id
    Generalisation_plot_path = generalisation_dir + "/" + id
    AUC_plot_path = AUC_plot_path + "/" + id

    long_data_file = ""
    short_data_file = ""
    test_data_path = ""

    results = RUN_PARTICIPANT(long_data_file, short_data_file, test_data_path, 
                    MagOrGrad, Filter_LowLimit, Filter_HighLimit, 
                    Apply_Baseline, Baseline_Window_Training, Developer_Mode,New_Sample_Freq, 
                    Crossval_acc_plot_path, Generalisation_plot_path, AUC_plot_path)
    
    all_participants_results.append(results)

Generate_Summary_Plots()
    

# STILL TO DO:
# Add the summary plots code  ##
# Add the filepaths 
# Work out nautilus file handling
# Ask Yunyun why AUC
# Test the code (MAKE SURE IT IS DOING LAST 2 SECONDS)
# Sanity check the model on left-out data (what left out data?)
# Possibly build a routine to combine all participant data for model training