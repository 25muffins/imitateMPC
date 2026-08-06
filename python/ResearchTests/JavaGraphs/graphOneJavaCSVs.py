import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('initialtest.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# trajectory
axes[0,0].plot(df['relX'], df['relY'])
axes[0,0].set_title('Trajectory')
axes[0,0].set_xlabel('x (in)')
axes[0,0].set_ylabel('y (in)')

# position error over time
axes[0,1].plot(df['time_s'], df['pos_error'])
axes[0,1].set_title('Position Error')
axes[0,1].set_xlabel('time (s)')
axes[0,1].set_ylabel('error (in)')

# cross-track error
axes[1,0].plot(df['time_s'], df['cross_track_error'])
axes[1,0].set_title('Cross-Track Error')
axes[1,0].set_xlabel('time (s)')
axes[1,0].set_ylabel('error (in)')

# theta error
axes[1,1].plot(df['time_s'], df['theta_error'])
axes[1,1].set_title('Heading Error')
axes[1,1].set_xlabel('time (s)')
axes[1,1].set_ylabel('error (rad)')

plt.tight_layout()
plt.savefig('metrics.pdf')