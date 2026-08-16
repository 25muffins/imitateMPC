import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set_context("paper")

models = {
    'Hard Switch':  'Hard_Full_Drawing.csv',
    'MLP':          'MLP_Full_Drawing.csv',
    'Sigmoid Gate': 'Sigmoid_Full_Drawing.csv',
    'Cross-Attn':   'Attention_Full_Drawing.csv',
    'AttentionGRU' : 'AttentionGRU_Full_Drawing.csv'
}

colors = ['steelblue', 'coral', 'green', 'purple', 'red']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for (name, path), color in zip(models.items(), colors):

    df = pd.read_csv(path)
    t  = df['time_s']
    # trajectory
    axes[0,0].plot(df['relX'], df['relY'], label=name, color=color)

    # position error
    axes[0,1].plot(t, df['pos_error'], label=name, color=color)

    # cross-track error
    axes[1,0].plot(t, df['cross_track_error'], label=name, color=color)

    # theta error
    axes[1,1].plot(t, df['theta_error'], label=name, color=color)

# mark goalswitch transition on all plots
df_ref = pd.read_csv(list(models.values())[0])
switch_t = df_ref[df_ref['goalswitch'] == 1]['time_s'].iloc[0]
# for ax in axes.flat:
#     ax.axvline(switch_t, color='gray', linestyle='--',
#                alpha=0.5, label='goal switch')

axes[0,0].tick_params(axis='both', labelsize=11)
axes[0,0].set_title('Trajectory')
axes[0,0].set_xlabel('x (in)')
axes[0,0].set_ylabel('y (in)')
axes[0,0].legend(fontsize=8)

axes[0,1].tick_params(axis='both', labelsize=11)
axes[0,1].set_title('Position Error')
axes[0,1].set_xlabel('time (s)')
axes[0,1].set_ylabel('error (in)')
axes[0,1].legend(fontsize=8)

axes[1,0].tick_params(axis='both', labelsize=11)
axes[1,0].set_title('Cross-Track Error')
axes[1,0].set_xlabel('time (s)')
axes[1,0].set_ylabel('error (in)')
axes[1,0].legend(fontsize=8)

axes[1,1].tick_params(axis='both', labelsize=11)
axes[1,1].set_title('Heading Error')
axes[1,1].set_xlabel('time (s)')
axes[1,1].set_ylabel('error (rad)')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('real_robot_comparison.pdf', bbox_inches='tight', dpi=150)
print("Saved: real_robot_comparison.pdf")

# summary table
print(f"\n{'Model':<15} {'MAE pos':>10} {'MAE cross':>10} {'MAE theta':>10}")
print("-" * 50)
for name, path in models.items():
    df  = pd.read_csv(path)
    print(f"{name:<15} "
          f"{df['pos_error'].mean():>10.3f} "
          f"{df['cross_track_error'].mean():>10.3f} "
          f"{df['theta_error'].mean():>10.3f}")