import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set_context("paper")

models1 = {
    'Hard Switch':  'Hard_Full_Drawing.csv',
    'MLP':          'MLP_Full_Drawing.csv',
    'Sigmoid Gate': 'Sigmoid_Full_Drawing.csv',
    'Cross-Attn':   'Attention_Full_Drawing.csv',
}
models2 = {
    'Hard Switch':  'Hard_Full_Drawing_jumpMag.csv',
    'MLP':          'MLP_Full_Drawing_jumpMag.csv',
    'Sigmoid Gate': 'Sigmoid_Full_Drawing_jumpMag.csv',
    'Cross-Attn':   'Attention_Full_Drawing_jumpMag.csv',
}

colors = ['steelblue', 'coral', 'green', 'purple']
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].set_aspect('auto')
axes[1].set_aspect('auto')

for (name, path), color in zip(models1.items(), colors):
    df = pd.read_csv(path)

    t  = df['time_s']
    axes[0].plot(t, df['pos_error'] * 0.0254, label=name, color=color)
for (name, path), color in zip(models2.items(), colors):
    df = pd.read_csv(path)
    t = df['time_s']
    axes[1].plot(t, df['jumpMag'] * 0.0254, label=name, color=color)

axes[0].axvline(0.825, color='gray', linestyle='--',
                alpha=0.7, label='goalswitch')
axes[0].tick_params(axis='both', labelsize=20)
axes[0].set_title('A. Position Error', fontsize=25)
axes[0].set_xlabel('time (s)')
axes[0].set_ylabel('error (m)')
axes[0].legend(fontsize=15)

axes[1].tick_params(axis='both', labelsize=20)
axes[1].set_title('B. Control jump during 200ms handoff interval ', fontsize=25)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('control jump (m/s)')



plt.tight_layout()
plt.savefig('PANEL_AB.pdf', bbox_inches='tight')