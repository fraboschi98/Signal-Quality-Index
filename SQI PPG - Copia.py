# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:54:38 2023

@author: FrancescaBoschi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import csv

def perfusion_evaluation(ppg,DC):
      
    peaks, _ = find_peaks(ppg, height=75, distance=16)  # 75 e 16  #5 250

    epochs = []
    max_indexes = []
    min_indexes = []
    AC_=[]
    min_values=[]



    for peak in peaks:
        start = peak - 16 #250 #16
        end = peak + 16 #250 #16
        if start < 0:
            start = 0
        if end > len(ppg):
            end = len(ppg)
        epoch = ppg[start:end]
        epochs.append(epoch)

        max_value = np.max(epoch)
        min_value = np.min(epoch)
        AC=max_value-min_value
        AC_.append(AC)
        min_values.append(min_value)
        
        
        max_index=np.argmax(epoch)
        min_index=np.argmin(epoch)
        max_indexes.append(max_index+start)
        min_indexes.append(min_index+start)
    
    #Cleaning
    ACmean=np.mean(AC_)
    ACdev=np.std(AC_)

    ACclean=[]
    for ac in AC_:
        if ac < ACmean-ACdev or ac > ACmean+ACdev:
            pass
        else:
            ACclean.append(ac)


    AC=np.mean(ACclean)
    
    
    
    DC=DC+np.mean(min_values)
    

    AC=np.mean(ACclean)
    perfusion=(AC/DC)*100
    
    font_properties_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 26}
    font_properties_axis_labels = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    font_properties_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
   
    
    plt.figure()
    plt.plot((ppg),label='Filtered PPG')
    plt.scatter(max_indexes, [ppg[i] for i in max_indexes], color='red', label='Max ')
    plt.scatter(min_indexes, [ppg[i] for i in min_indexes], color='green', label='Min ')
    plt.xlabel('Samples', **font_properties_axis_labels)
    plt.ylabel('PPG (LSB)', **font_properties_axis_labels)
    plt.title('PPG SQI: Perfusion', **font_properties_title)

    plt.legend(prop=font_properties_legend)

    plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
    plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])

    plt.legend()
    plt.show()
   
    return epochs, perfusion
    

def pc_evaluation(epochs):
    correlation_coefficients = []

    for i in range(len(epochs)-1):
        epoch1 = epochs[i]
        epoch2 = epochs[i+1]
        
        if len(epoch1) >= 2 and len(epoch2) >= 2:  # Check if both epochs have length >= 2
            min_length = min(len(epoch1), len(epoch2))
            epoch1 = epoch1[:min_length]
            epoch2 = epoch2[:min_length]

            correlation, _ = pearsonr(epoch1, epoch2)
            correlation_coefficients.append(correlation)
    return correlation_coefficients

def baseline_removal(ecg_signal, window_size):
    ecg_signal=ecg_signal.flatten()
    
    smoothed_signal = np.convolve(ecg_signal, np.ones(window_size)/window_size, mode='same')
    
    
    baseline_removed_signal = ecg_signal - smoothed_signal
    
    return baseline_removed_signal

#----------------Signal Loading----------------#
font_properties_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 26}
font_properties_axis_labels = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font_properties_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}

fs_ecg=128  # Sampling frequency (Hz)
fs_ppg=32  # Sampling frequency (Hz)
#ECG
ecg_mat=scipy.io.loadmat('SINTEC6ECG.mat')
ecg=ecg_mat['signal']
tsecg=ecg_mat['ts']

#PPG
ppg_mat=scipy.io.loadmat('SINTEC6PPG.mat')
ppg=ppg_mat['signal']
tsppg=ppg_mat['ts']
ppgfil = baseline_removal(ppg, int(np.round(fs_ppg/2)))


csv_file_name = "SINTEC6.csv"

ecg_head = np.array(ecg).flatten()
ts_ecg= np.array(tsecg).flatten()
ppg_head = np.array(ppg).flatten()
ts_ppg= np.array(tsppg).flatten()

#Reference device
with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    ts_omron = np.array(list(map(float, [(line[0]) for line in reader])))

with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    sbp = np.array(list(map(float, [(line[1]) for line in reader])))

with open(csv_file_name) as filecsv:
    reader = csv.reader(filecsv, delimiter=";")
    dbp = np.array(list(map(float, [(line[2]) for line in reader])))


# Reference device's time values correspond to the time in which the device returns the pressure values
ts_omron=ts_omron-60*np.ones(len(ts_omron)) 
 







rec_time_mins_ppg = ((len(ppg_head)-1)/fs_ppg)/60 
rec_time_mins_ecg = ((len(ecg_head)-1)/fs_ecg)/60



#=================
# SIGNAL PREPARING
#=================
# Signal alignment 
tstart=ts_omron[0]; #timestamp of the start sample of protocol registration

positive_diff_ppg = ts_ppg - tstart
positive_diff_ecg = ts_ecg - tstart


idx_ppg_start = np.where(positive_diff_ppg >= 0)[0][np.argmin(positive_diff_ppg[positive_diff_ppg >= 0])] #the index of the PPG sample with the least positive difference with the tstart sample is determined
idx_ecg_start = np.where(positive_diff_ecg >= 0)[0][np.argmin(positive_diff_ecg[positive_diff_ecg >= 0])] #the index of the ECG sample with the least positive difference with the tstart sample is determined

# Signal cut
idx_ppg_end=int(np.round(20*60*fs_ppg))+idx_ppg_start
ppg_head=ppg_head[idx_ppg_start:idx_ppg_end] 
ppgfil=ppgfil[idx_ppg_start:idx_ppg_end]
ts_ppg=ts_ppg[idx_ppg_start:idx_ppg_end]

idx_ecg_end=int(np.round(20*60*fs_ecg))+idx_ecg_start
ecg_head=ecg_head[idx_ecg_start:idx_ecg_end] 
ts_ecg=ts_ecg[idx_ecg_start:idx_ecg_end]


t_ppg = np.arange(0,len(ppg_head))/fs_ppg
t_ecg = np.arange(0,len(ecg_head))/fs_ecg


fs=32
ppg=ppg_head
time=t_ppg


#------------------ Settings ------------------#

nfft = 1024
noverlap = 16 
window = signal.hamming(32)  
  

cutoff_freq = 0.1 #cutoff frequency for BW evaluation
normalized_cutoff = cutoff_freq / (fs / 2)

#------------------ PSD ------------------#
frequencies, psd = signal.welch(ppg-np.mean(ppg), fs=fs, window=window, noverlap=noverlap, nfft=nfft)




#------------------ PSD PPG ------------------#
freq_range_1_2_25 = (frequencies >= 1) & (frequencies <= 2.25)
freq_range_0_8 = (frequencies >= 1) & (frequencies <= 8)

psd_range1 = psd[freq_range_1_2_25]
psd_range2 = psd[freq_range_0_8]

ratio_PPG = np.sum(psd_range1) / np.sum(psd_range2)


print("PPG PSD ratio: {:.1f}".format(ratio_PPG))



#------------------ BW ------------------#
#IIR LP filter
b, a = butter(4, normalized_cutoff, btype='high', analog=False)
filtered_bw = filtfilt(b, a, ppg)
#f,lowpsd=signal.welch(filtered_bw-np.mean(filtered_bw), fs=fs, window=window, noverlap=noverlap, nfft=nfft)
freq_range_0_01 = (frequencies > 0) & (frequencies <= 0.1)
freq_range_0_40 = (frequencies > 0) & (frequencies <= 40)


psd_range_0_01 = psd[freq_range_0_01]
psd_range_0_40 = psd[freq_range_0_40]




ratio_bw = np.sum(psd_range_0_01) / np.sum(psd_range_0_40)
baseline_ratio_bw=1-ratio_bw

# Q index
I_bw=np.max(ppg)-np.min(ppg)
I_rbw=np.max(filtered_bw)-np.min(filtered_bw)
Qbw=1/(1+I_bw/I_rbw)



if baseline_ratio_bw > 0.5:
    print("The relative power in baseline shows a good signal quality")
else:
    print("The relative power in baseline shows a bad signal quality")
print("The relative power in baseline is: {:.1f}".format(baseline_ratio_bw))
print("The baseline wander influencing degree is: {:.2f}".format(Qbw))

b, a = butter(4, normalized_cutoff, btype='low', analog=False)
filtered_bw = filtfilt(b, a, ppg)
t = (np.arange(0,len(ppg))/fs)
epochstart=int(np.round(0*60*fs))
epochend=int(np.round(20*60*fs))

plt.figure()
plt.subplot(211)
plt.plot(t[epochstart:epochend],ppg[epochstart:epochend],label='PPG signal')
plt.plot(t[epochstart:epochend],filtered_bw[epochstart:epochend],label='baseline wander',color='red',linewidth=3.0)

plt.title('PPG signal baseline wander ', fontsize=32, fontname='Times New Roman',fontweight='bold')
plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('PPG (LSB)', **font_properties_axis_labels)
plt.title('PPG SQI: Baseline wander', **font_properties_title)

plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])

plt.subplot(212)
plt.plot(t[epochstart:epochend],ppg[epochstart:epochend],label='PPG signal')
plt.plot(t[epochstart:epochend],filtered_bw[epochstart:epochend],label='baseline wander',color='red',linewidth=3.0)


plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('PPG (LSB)', **font_properties_axis_labels)

plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.show()

#------------------ Perfusion ------------------#
DC=np.mean(filtered_bw)
epochs,perfusion=perfusion_evaluation(ppgfil,DC)




if perfusion >=2:
    print("PPG perfusion SQI shows a good signal quality")
else:
    print("PPG perfusion SQI shows a bad signal quality")
print("Average perfusion: {:.1f}".format(perfusion))

#------------------Coeff------------------------#
correlation_coefficients=pc_evaluation(epochs)
average_correlation = round(np.mean(correlation_coefficients),1)
if average_correlation >=0.5:
    print("PPG Pearson coefficient  SQI shows a good signal quality")
else:
    print("PPG Pearson coefficient  SQI shows a bad signal quality")
print("Average Pearson coefficient : {:.1f}".format(average_correlation))


plt.figure()

plt.plot(frequencies, psd)
plt.axvline(0.1,color='red', label='0.1 Hz')
plt.axvline(1,color='blue', label='1 Hz')
plt.axvline(2.5,color='green', label='2.5 Hz')
plt.axvline(8,color='orange', label='8 Hz')


plt.xlabel('Frequency (Hz)', **font_properties_axis_labels)
plt.ylabel('PSD PPG ', **font_properties_axis_labels)
plt.title('PPG Power Spectral Density', **font_properties_title)


plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.show()








