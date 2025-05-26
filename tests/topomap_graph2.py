import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# 1. 전극 채널 위치 정보 (예시)
channel_pos = {
    'Fp1': (-0.35, 1.0),
    'Fp2': (0.35, 1.0),
    'F7': (-0.85, 0.6),
    'F8': (0.85, 0.6),
    'F3': (-0.45, 0.5),
    'Fz': (0.0, 0.45),
    'F4': (0.45, 0.5),
    'T3': (-1.0, 0.0), 
    'C3': (-0.55, 0.0),
    'Cz': (0.0, 0.0),
    'C4': (0.55, 0.0),
    'T4': (1.0, 0.0), 
    'T5': (-0.85, -0.6),
    'P3': (-0.45, -0.5),
    'Pz': (0.0, -0.45),
    'P4': (0.45, -0.5),
    'T6': (0.85, -0.6),
    'O1': (-0.35, -1.0),
    'O2': (0.35, -1.0),
}

# 3. 연결할 채널 쌍 (엣지)
'''
edges = [('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
         ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
         ('C3', 'Cz'), ('Cz', 'C4')]
'''

channels = [
    'Fp1',
    'Fp2',
    'F7',
    'F8',
    'F3',
    'Fz',
    'F4',
    'C3',
    'Cz',
    'C4',
    'T3', 
    'T4', 
    'T5',
    'T6', 
    'P3',
    'Pz',
    'P4',
    'O1',
    'O2',
]

channels_li = []
channels_li.append(channels[:7])
channels_li.append(channels[14:17])
channels_li.append(channels[10:14])
channels_li.append(channels[7:10])
channels_li.append(channels[17:])

edges_li = []
for channels in channels_li:
    edges = list(combinations(channels, 2))
    edges_li.append(edges)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# 채널 점과 값 그리기
colors_li = ['firebrick', 'chocolate', 'green', 'steelblue', 'slateblue']
for idx, channels in enumerate(channels_li):
    for ch in channels:
        x, y = channel_pos[ch]
        ax.scatter(x, y, s=1000, zorder=3, c=colors_li[idx])
        ax.text(x, y, ch, fontsize=15, ha='center', va='center', zorder=4, color='white')

# 엣지 그리기
colors_li = ['lightcoral', 'lightsalmon', 'lightgreen', 'cornflowerblue', 'mediumpurple']
for idx, edges in enumerate(edges_li, start=0):
    for ch1, ch2 in edges:
        x1, y1 = channel_pos[ch1]
        x2, y2 = channel_pos[ch2]
        ax.plot([x1, x2], [y1, y2], colors_li[idx], linewidth=2, zorder=2)

# 배경 원 그리기 (두피)
circle = plt.Circle((0, 0), 1.3, color='black', fill=False, linewidth=2, zorder=1)
ax.add_artist(circle)

# 좌측 귀 (반원)
theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)  # 왼쪽 반원
ear_left_x = -1.28 + 0.2 * np.cos(theta)
ear_left_y = 0.0 + 0.3 * np.sin(theta)
ax.plot(ear_left_x, ear_left_y, color='black', linewidth=2, zorder=1)

# 우측 귀 (반원)
theta = np.linspace(-np.pi / 2, np.pi / 2, 100)  # 오른쪽 반원
ear_right_x = 1.28 + 0.2 * np.cos(theta)
ear_right_y = 0.0 + 0.3 * np.sin(theta)
ax.plot(ear_right_x, ear_right_y, color='black', linewidth=2, zorder=1)

# 코 (아래 방향 삼각형 - 밑면 생략, 위로 이동)
nose_tip_y = 1.45
nose_base_y = 1.3
nose_x = [0.0, -0.1, 0.1]
nose_y = [nose_tip_y, nose_base_y, nose_base_y]

# 꼭짓점에서 왼쪽 점, 꼭짓점에서 오른쪽 점만 그림
ax.plot([nose_x[0], nose_x[1]], [nose_y[0], nose_y[1]], color='black', linewidth=2, zorder=1)
ax.plot([nose_x[0], nose_x[2]], [nose_y[0], nose_y[2]], color='black', linewidth=2, zorder=1)

ax.set_aspect('equal')

LIMIT = 1.6
ax.set_xlim(-LIMIT, LIMIT)
ax.set_ylim(-LIMIT, LIMIT)
ax.axis('off')
plt.title('Ours', size=20)
plt.savefig('./assets/tomomap_2.jpg', dpi=500)
plt.show()