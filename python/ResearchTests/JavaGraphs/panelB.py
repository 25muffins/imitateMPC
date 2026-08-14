import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
sns.set_context("paper")

models = {
    'Hard ':  'Hard_Full_Drawing_jumpMag.csv',
    'MLP':          'MLP_Full_Drawing_jumpMag.csv',
    'Scalar Gate': 'Sigmoid_Full_Drawing_jumpMag.csv',
    'Lightweight Attn':   'Attention_Full_Drawing_jumpMag.csv',
}

colors = ['steelblue', 'coral', 'green', 'purple']
fig, axes = plt.subplots(1, 1, figsize=(8, 8))

for (name, path), color in zip(models.items(), colors):
    df = pd.read_csv(path)

    t  = df['time_s']
    axes.plot(t, df['jumpMag'] * 0.0254, label=name, color=color)

axes.tick_params(axis='both', labelsize=25)
axes.set_title('B. Control jump during 200ms handoff interval ', fontsize=32)
axes.set_xlabel('time (s)', fontsize=25)
axes.set_ylabel('control jump (m/s)', fontsize=25)
axes.legend(fontsize=17)
plt.tight_layout()

plt.tight_layout()
plt.savefig('PANEL_B.pdf', bbox_inches='tight', dpi=150)