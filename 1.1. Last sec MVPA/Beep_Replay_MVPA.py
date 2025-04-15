# %% Parameters

# Developer Mode
Developer_Mode = True
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

# Beep Time Sequences
long1 = [0.8, 1.2, 1.6]
long2 = [0.8, 1.6, 1.2]
long3 = [1.2, 0.8, 1.6]
long4 = [1.2, 1.6, 0.808]
long5 = [1.6, 0.8, 1.2]
long6 = [1.6, 1.2, 0.8]

medium1 = [0.5333, 0.8, 1.0667]
medium2 = [0.536, 1.064, 0.8]
medium3 = [0.803, 0.532, 1.064]
medium4 = [0.799, 1.068, 0.532]
medium5 = [1.064, 0.536, 0.8]
medium6 = [1.064, 0.8, 0.536]

short1 = [0.3556, 0.5333, 0.7111]
short2 = [0.356, 0.712, 0.532]
short3 = [0.532, 0.356, 0.712]
short4 = [0.532, 0.712, 0.356]
short5 = [0.708, 0.3559, 0.536]
short6 = [0.712, 0.5359, 0.352]

Beep_Time_Sequences = [long1, long2, long3, long4, long5, long6,
                       medium1, medium2, medium3, medium4, medium5, medium6, 
                       short1, short2, short3, short4, short5, short6]

# Graph Output Directories
#plot_dir = "home/jm283064/"
#crossval_decoding_dir = plot_dir + "CrossValDecodingAccuracy"
#generalisation_dir = plot_dir + "GeneralisationMatrix"
#testauc_dir = plot_dir + "TestAUC"
#for d in [crossval_decoding_dir, generalisation_dir, testauc_dir]:
#    os.makedirs(d, exist_ok=True)


# %% Subroutines

# %% Main Program

for id in participant_ids:
    for seq in Beep_Time_Sequences:

        Beep_Epochs = Preprocess_Beeps(seq, Full_Epoch_File_Path,
                                                                   Apply_Baseline, Baseline_Window_Training,
                                                                   Developer_Mode)
        
        Train_Beep_Epochs, Validate_Beep_Epochs = Split_Beeps_LeftOut_Val(Beep_Epochs),

        

"""

THINK ABOUT: YOUR BEEP EPOCHS NEED TO INCLUDE NON-BEEP DATA FOR THE CLASSIFIER. HOW BIG SHOULD THE BEEP WINDOW BE??

Plan:

1. Find out when to expect the beep and what size iterval to use i.e. how much lag
2. Extract beep intervals, chop from each epoch and make into its own array
    Do not use 1-item trials! As they will be used for validation and generalisation.
    Do not use final beep! As the screen changes to orange. 
3. Within each group of beep intervals, remove 12.5% of trials for sanity check.
4. Perform the sanity check.
5. Perform sanity check on ending period of 1-item trials (first beep). Maybe for each condition.
6. Plot a heatmap for EACH single item condition (long and short)
7. Plot a heatmap for the 12.5% left out data too, for EACH left out condition.
8. Summary plots for all participants
9. Combine all data into one large participant and try the above again.


This week:
    Do the one-item analyses. Sanity checks on left-out data... Be precise what you did on slides for write up
    Do the beep method (this one)
    Try CSP
    Plot spectral power analyses
    Read Sophie's paper
    Extention: Source reconstruction



"""