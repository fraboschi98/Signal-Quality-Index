# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:49:14 2023

@author: FrancescaBoschi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import kurtosis
import csv
from scipy.signal import find_peaks
from matplotlib import rcParams


font_properties_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 26}
font_properties_axis_labels = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font_properties_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}


def kurtosis_evaluation(ecg, epoch_samples):
    kSQI_ecg = []
    total_epochs = int(len(ecg) // epoch_samples)

    for i in range(total_epochs):
        epoch_start = int(i * epoch_samples)
        epoch_end = int((i + 1) * epoch_samples)
        epoch = ecg[epoch_start:epoch_end]

        epoch_kurtosis = kurtosis(epoch)
        kSQI_ecg.append(epoch_kurtosis)

    epoch_length = int(epoch_samples)
    kSQI_ecg_repeated = []

    for value in kSQI_ecg:
        repeated_values = [value] * epoch_length
        kSQI_ecg_repeated.extend(repeated_values)

    last_samples = len(kSQI_ecg_repeated)
    ecg_cut = ecg[:last_samples]
    mean_kurtosis = np.mean(kSQI_ecg_repeated)
    return kSQI_ecg_repeated, mean_kurtosis,ecg_cut

#----------------Signal Loading----------------#
#ECG
ecg_mat=scipy.io.loadmat('SHIMMER_H8_ECG.mat')
ecg=ecg_mat['signal']
tsecg=ecg_mat['ts']

#PPG
ppg_mat=scipy.io.loadmat('SHIMMER_H8_PPG.mat')
ppg=ppg_mat['signal']
tsppg=ppg_mat['ts']

#Omron
csv_file_name = "SHIMMER_H8.csv"

fs_ecg=504.12  # Sampling frequency (Hz)
fs_ppg=504.12  # Sampling frequency (Hz)


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
idx_ppg_end = int(np.round(20 * 60 * fs_ppg)) + idx_ppg_start
ppg_head = ppg_head[idx_ppg_start:idx_ppg_end]
ts_ppg=ts_ppg[idx_ppg_start:idx_ppg_end]

idx_ecg_end=int(np.round(20*60*fs_ecg))+idx_ecg_start
ecg_head=ecg_head[idx_ecg_start:idx_ecg_end] 
ts_ecg=ts_ecg[idx_ecg_start:idx_ecg_end]


t_ppg = np.arange(0,len(ppg_head))/fs_ppg
t_ecg = np.arange(0,len(ecg_head))/fs_ecg

fs=504.12
ecg=ecg_head
time=t_ecg
#------------------ Settings ------------------#

epoch_duration=1 # for kurtosis evaluation on  epochs
epoch_samples=epoch_duration*fs
kurtosis_th=5 #above 5 good ecg, gaussian noise around 3

nfft = 1024
noverlap = 64 
window = signal.hamming(128)  

cutoff_freq = 0.1 #cutoff frequency for BW evaluation
normalized_cutoff = cutoff_freq / (fs / 2)

powerline_freq = 50  #Notch cutoff frequency for PLI evaluation
notch_freq = powerline_freq / (fs / 2) 
Q=30 # Quality factor for PLI evaluation


#------------------ PSD ------------------#
frequencies, psd = signal.welch(ecg-np.mean(ecg), fs=fs, window=window, noverlap=noverlap, nfft=nfft)


#------------------ Kurtosis ------------------#
kSQI_ecg_repeated,mean_kurtosis,ecg_cut=kurtosis_evaluation(ecg, epoch_samples)
t = (np.arange(0,len(ecg_cut))/fs)/60
if mean_kurtosis >= 5:
    print("ECG kurtosis SQI shows a good signal quality")
else:
    print("ECG kurtosis SQI shows a bad signal quality")
print("Average kurtosis: {:.1f}".format(mean_kurtosis))

plt.figure()
mask = np.array(kSQI_ecg_repeated) < 5

plt.plot(t*60,ecg_cut,  label='Kurtosis > 5')

plt.plot(t*60,np.where(mask, ecg_cut, np.nan), color='red', label='Kurtosis < 5')

plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('ECG (mV)', **font_properties_axis_labels)
plt.title('ECG SQI: Kurtosis', **font_properties_title)

plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])

plt.show()

#------------------ PSD QRS ------------------#
freq_range_5_15 = (frequencies >= 5) & (frequencies <= 15)
freq_range_5_40 = (frequencies >= 5) & (frequencies <= 40)

psd_range_5_15 = psd[freq_range_5_15]
psd_range_5_40 = psd[freq_range_5_40]

ratio_QRS = np.sum(psd_range_5_15) / np.sum(psd_range_5_40)

if ratio_QRS >= 0.5 and ratio_QRS <= 0.8:
    print("ECG QRS SQI shows a good signal quality")
else:
    print("ECG QRS SQI shows a bad signal quality")
print("QRS PSD ratio: {:.1f}".format(ratio_QRS))

#------------------ BW ------------------#

#IIR LP filter
b, a = butter(4, normalized_cutoff, btype='high', analog=False)
filtered_bw = filtfilt(b, a, ecg)

#f,lowpsd=signal.welch(filtered_bw-np.mean(filtered_bw), fs=fs, window=window, noverlap=noverlap, nfft=nfft)


freq_range_0_01 = (frequencies > 0) & (frequencies <= 0.1)
freq_range_0_40 = (frequencies > 0) & (frequencies <= 40)


psd_range_0_01 = psd[freq_range_0_01]
psd_range_0_40 = psd[freq_range_0_40]
# Q index
I_bw=np.max(ecg)-np.min(ecg)
I_rbw=np.max(filtered_bw)-np.min(filtered_bw)
Qbw=1/(1+I_bw/I_rbw)




ratio_bw = np.sum(psd_range_0_01) / np.sum(psd_range_0_40)
baseline_ratio_bw=1-ratio_bw
if baseline_ratio_bw > 0.5:
    print("The relative power in baseline shows a good signal quality")
else:
    print("The relative power in baseline shows a bad signal quality")
print("The relative power in baseline is: {:.2f}".format(baseline_ratio_bw))
print("The baseline wander influencing degree is: {:.2f}".format(Qbw))

b, a = butter(4, normalized_cutoff, btype='low', analog=False)
filtered_bw = filtfilt(b, a, ecg)

t = (np.arange(0,len(ecg))/fs)
epochstart=int(np.round(0*60*fs))
epochend=int(round(20*60*fs))
plt.figure()
plt.subplot(211)
plt.plot(t[epochstart:epochend],ecg[epochstart:epochend],label='original ECG')
plt.plot(t[epochstart:epochend],filtered_bw[epochstart:epochend],label='baseline wander',color='red',linewidth=3.0)

plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('ECG (mV)', **font_properties_axis_labels)
plt.title('ECG SQI: Baseline wander', **font_properties_title)

plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])


plt.subplot(212)
plt.plot(t[epochstart:epochend],ecg[epochstart:epochend],label='original ECG')
plt.plot(t[epochstart:epochend],filtered_bw[epochstart:epochend],label='baseline wander',color='red',linewidth=3.0)

plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('ECG (mV)', **font_properties_axis_labels)


plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])



plt.show()

#------------------ PLI ------------------#
#Notch filter
b, a = signal.iirnotch(notch_freq, Q)
filtered_notch = signal.lfilter(b, a, ecg)

PLI=ecg-filtered_notch
frequencies_PLI, psd_PLI = signal.welch(PLI, fs=fs, window=window, noverlap=noverlap, nfft=nfft)



#------------------ Figures ------------------#
plt.figure()
plt.subplot(211)
plt.plot(time,ecg,label='ECG')
plt.plot(time,PLI,label='PLI',color='red')

plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('ECG (mV)', **font_properties_axis_labels)
plt.title('ECG SQI: PLI', **font_properties_title)

plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])


plt.subplot(212)
plt.plot(time,ecg,label='ECG')
plt.plot(time,PLI,label='PLI',color='red')

plt.xlabel('Time (seconds)', **font_properties_axis_labels)
plt.ylabel('ECG (mV)', **font_properties_axis_labels)


plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])






plt.show()

plt.figure()

plt.plot(frequencies, psd)
plt.axvline(0.1,color='red', label='0.1 Hz')
plt.axvline(5,color='blue', label='5 Hz')
plt.axvline(15,color='green', label='15 Hz')
plt.axvline(40,color='orange', label='40 Hz')


plt.xlabel('Frequency (Hz)', **font_properties_axis_labels)
plt.ylabel('PSD ECG ', **font_properties_axis_labels)
plt.title('ECG Power Spectral Density', **font_properties_title)


plt.legend(prop=font_properties_legend)

plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.show()


plt.figure()

plt.plot(frequencies_PLI, psd_PLI)



plt.xlabel('Frequency (Hz)', **font_properties_axis_labels)
plt.ylabel('PSD PLI ', **font_properties_axis_labels)
plt.title('PLI Power Spectral Density', **font_properties_title)




plt.xticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])
plt.yticks(fontsize=font_properties_axis_labels['size'], family=font_properties_axis_labels['family'])



plt.show()











