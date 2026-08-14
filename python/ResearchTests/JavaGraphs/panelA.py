import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper")

models = {
    'Hard Switch':  'Hard_Full_Drawing.csv',
    'MLP':          'MLP_Full_Drawing.csv',
    'Sigmoid Gate': 'Sigmoid_Full_Drawing.csv',
    'Cross-Attn':   'Attention_Full_Drawing.csv',
}

colors = ['steelblue', 'coral', 'green', 'purple']
fig, axes = plt.subplots(1, 1, figsize=(8, 8))

for (name, path), color in zip(models.items(), colors):
    df = pd.read_csv(path)

    t  = df['time_s']
    axes.plot(t, df['pos_error'] * 0.0254, label=name, color=color)

axes.axvline(0.825, color='gray', linestyle='--',
                alpha=1.0, label='goalswitch')
axes.tick_params(axis='both', labelsize=25)
axes.set_title('A. Position Error', fontsize=32)
axes.set_xlabel('time (s)', fontsize=25)
axes.set_ylabel('error (m)', fontsize=25)
axes.legend(fontsize=17, loc = 'upper right')
plt.tight_layout()
plt.savefig('PANEL_A.pdf', bbox_inches='tight', dpi=150)