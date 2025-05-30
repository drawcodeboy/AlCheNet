import sys, os
sys.path.append(os.getcwd())
import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_eeglab(r'data\open-nuro-dataset\dataset\sub-001\eeg\sub-001_task-eyesclosed_eeg.set', preload=False)

data, times = raw[:19]  # 예: 처음 5개 채널 데이터, shape (5, n_times)
channels = raw.ch_names[:19]
ran = 1000
plt.figure(figsize=(6, 12))
for i, ch_data in enumerate(data):
    plt.plot(times[1000:3000], ch_data[1000:3000] * 1e7 + i*ran, label=channels[i])  # µV 단위로 변환 + 오프셋
plt.yticks([i*ran for i in range(19)], channels)
plt.xlabel('Time (s)')
plt.ylabel('Channels')
plt.title('EEG signals with grid')
plt.grid(True)
plt.show()
# plt.savefig("assets/eeg_sample.jpg", dpi=500)