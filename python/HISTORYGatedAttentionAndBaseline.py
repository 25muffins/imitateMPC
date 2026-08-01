import math

import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

matlab_files = ['HistoryDataV1.mat',
               'HistoryDataV2.mat']

ALPHA = 0.05 #this is just so softmax doesn't equal 1,0
VAL_PERCENT = 0.15 #15% validation data
HIDDEN_NEURONS = 64
EPOCHS        = 120
BATCH_SIZE    = 512
LR_INITIAL    = 1e-3
LR_DROP_EPOCH = 50       # same as MATLAB
LAMBDA_ATTN   = 0.01     # weight of attention supervision loss
SAVE_PATH = 'il_mpc_attention.tflite'

def load_data(mat_files):
    parts = []
    for f in mat_files:
        mat = scipy.io.loadmat(f)
        parts.append(mat['d'])
    data = np.vstack(parts).astype(np.float64)
    print(f"shape: {data.shape}")
    return data


def extract_features(data):
    x_y = data[:, [0, 1]]  # x, y
    sin_cos_th = data[:, [32, 33]]  # sin(theta), cos(theta)
    vel = data[:, [3, 4, 5]]  # vx, vy, omega

    goal1_xy = data[:, [6, 7]]  # goal1 x, y
    sin_cos_g1 = data[:, [34, 35]]  # sin/cos goal1 heading

    goal2_xy = data[:, [14, 15]]  # goal2 x, y
    sin_cos_g2 = data[:, [36, 37]]  # sin/cos goal2 heading

    # probably dont need
    past_u = data[:, [25, 26, 27]]  # past xvel, yvel, omega

    goalswitch = data[:, [28]]  #goalswitch
    target = data[:, [22, 23, 24]]  # ideal vx, vy, omega

    query = np.concatenate([x_y, sin_cos_th, vel, goalswitch], axis=1)
    goal1 = np.concatenate([goal1_xy, sin_cos_g1], axis=1)
    goal2= np.concatenate([goal2_xy, sin_cos_g2], axis=1)
    waypoints = np.stack([goal1, goal2], axis=1)

    return  query, waypoints, past_u, goalswitch, target

def normalize(currentState, waypoints, past_u, target):
    state_norm = (currentState / np.array([72, 72, 1, 1, 30, 30, np.pi, 1])).astype(np.float32)
    wp_norm = (waypoints / np.array([72, 72, 1, 1])).astype(np.float32)
    past_norm = (past_u / np.array([30 * math.sqrt(1), 30* math.sqrt(1), np.pi])).astype(np.float32)
    target_norm = (target / np.array([30* math.sqrt(1), 30* math.sqrt(1), np.pi])).astype(np.float32)

    return state_norm, wp_norm, past_norm, target_norm


def make_attention_targets(goalswitch):

    gs = goalswitch.ravel()
    targets = np.where(
        gs[:, None] == 0,
        [[1 - ALPHA, ALPHA]],
        [[ALPHA, 1 - ALPHA]]
    ).astype(np.float32)
    return targets  # (N, 2)



def temporal_split(query_norm, wp_norm, target_norm,
                   attn_targets, goalswitch):

    N = len(query_norm)
    split = int((1 - VAL_PERCENT) * N)

    Xq_tr, Xq_val = query_norm[:split], query_norm[split:]
    Xw_tr, Xw_val = wp_norm[:split], wp_norm[split:]
    Y_tr, Y_val = target_norm[:split], target_norm[split:]
    Yat_tr = attn_targets[:split]
    gs_val = goalswitch[split:]

    print(f"Train: {len(Xq_tr)}  Val: {len(Xq_val)}")
    return Xq_tr, Xq_val, Xw_tr, Xw_val, Y_tr, Y_val, Yat_tr, gs_val
def random_split(query_norm, wp_norm, target_norm,
                 attn_targets, goalswitch,
                 test_frac=0.05):

    N = len(query_norm)
    idx = np.random.permutation(N)

    n_val  = int(VAL_PERCENT  * N)
    n_test = int(test_frac * N)

    val_idx  = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    Xq_tr,   Xq_val  = query_norm[train_idx],  query_norm[val_idx]
    Xw_tr,   Xw_val  = wp_norm[train_idx],     wp_norm[val_idx]
    Y_tr,    Y_val   = target_norm[train_idx],  target_norm[val_idx]
    Yat_tr           = attn_targets[train_idx]
    gs_val           = goalswitch[val_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  "
          f"Test: {n_test}")
    return (Xq_tr, Xq_val, Xw_tr, Xw_val,
            Y_tr, Y_val, Yat_tr, gs_val)

class ILMPCAttentionPolicy(tf.keras.Model):
    def __init__(self, hidden=HIDDEN_NEURONS):
        super().__init__()
        self.query_enc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu',
                                  input_shape=(8,))
        ], name='query_enc')

        self.wp_enc = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden, activation='relu'),
            name='wp_enc'
        )

        self.W_q = tf.keras.layers.Dense(hidden, use_bias=False,
                                         name='W_q')
        self.W_k = tf.keras.layers.Dense(hidden, use_bias=False,
                                         name='W_k')
        self.W_v = tf.keras.layers.Dense(hidden, use_bias=False,
                                         name='W_v')
        self.scale = tf.math.sqrt(tf.cast(hidden, tf.float32))

        self.fc1 = tf.keras.layers.Dense(hidden, activation='relu',
                                         name='fc1')
        self.fc2 = tf.keras.layers.Dense(3, name='fc2')
        self.out_act = tf.keras.layers.Activation('tanh', name='tanh')

    def call(self, inputs, return_weights=False):
        query, waypoints = inputs
        # query:     (B, 8)
        # waypoints: (B, 2, 4)

        # Encode
        q_enc = self.query_enc(query)  # (B, hidden)
        kv = self.wp_enc(waypoints)  # (B, 2, hidden)

        # Project
        Q = tf.expand_dims(self.W_q(q_enc), 1)  # (B, 1, hidden)
        K = self.W_k(kv)  # (B, 2, hidden)
        V = self.W_v(kv)  # (B, 2, hidden)

        # Attend
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)  # (B, 1, 2)
        context = tf.squeeze(
            tf.matmul(weights, V), axis=1
        )  # (B, hidden)

        # Output
        combined = tf.concat([q_enc, context], axis=-1)  # (B, 2*hidden)
        out = self.out_act(self.fc2(self.fc1(combined)))  # (B, 3)

        attn_w = tf.squeeze(weights, axis=1)  # (B, 2)

        if return_weights:
            return out, attn_w
        return out, attn_w


@tf.function
def train_step(model, optimizer, Xq, Xw, Y, Yat):
    with tf.GradientTape() as tape:
        pred, weights = model([Xq, Xw], return_weights=True)
        err = tf.abs(pred - Y)
        loss_track = (
                1.0 * tf.reduce_mean(err[:, 0]) +
                1.0 * tf.reduce_mean(err[:, 1]) +
                20.0 * tf.reduce_mean(err[:, 2])  # omega emphasized
        )
        # KL divergence attention supervision
        w_clip = tf.clip_by_value(weights, 1e-8, 1.0)
        loss_attn = tf.reduce_mean(
            tf.reduce_sum(
                Yat * tf.math.log(Yat / w_clip), axis=-1
            )
        )

        total = loss_track + LAMBDA_ATTN * loss_attn

    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_track, loss_attn


def train(model, Xq_tr, Xw_tr, Y_tr, Yat_tr,
          Xq_val, Xw_val, Y_val, gs_val):
    optimizer = tf.keras.optimizers.Adam(LR_INITIAL, epsilon=1e-8)
    history = {'loss_track': [], 'loss_attn': [],
               'val_mae': [], 'attn_acc': []}
    best_mae = np.inf
    patience = 0
    PATIENCE = 20

    print(f"\n{'Epoch':>6} | {'track':>8} | {'attn':>8} | "
          f"{'val_mae':>8} | {'attn_acc':>9} | weights")
    print("-" * 72)

    for epoch in range(EPOCHS):
        if epoch == LR_DROP_EPOCH:
            optimizer.learning_rate.assign(LR_INITIAL * 0.5)
            print(f"  LR -> {LR_INITIAL * 0.5}")

        idx = np.random.permutation(len(Xq_tr))
        t_lst, a_lst = [], []

        for i in range(0, len(idx), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            lt, la = train_step(
                model, optimizer,
                Xq_tr[b], Xw_tr[b], Y_tr[b], Yat_tr[b]
            )
            t_lst.append(float(lt))
            a_lst.append(float(la))

        # Validation
        vp, vw = model([Xq_val, Xw_val], return_weights=True)
        vp, vw = vp.numpy(), vw.numpy()

        err = np.abs(vp - Y_val)
        val_mae = (1.0 * err[:, 0].mean() + 1.0 * err[:, 1].mean() +
                   20.0 * err[:, 2].mean())

        pred_g = np.argmax(vw, axis=1)
        true_g = (gs_val.ravel() > 0.5).astype(int)
        attn_acc = (pred_g == true_g).mean()
        mean_w = vw.mean(axis=0)

        lt_m = np.mean(t_lst)
        la_m = np.mean(a_lst)
        history['loss_track'].append(lt_m)
        history['loss_attn'].append(la_m)
        history['val_mae'].append(val_mae)
        history['attn_acc'].append(attn_acc)

        if epoch % 10 == 0:
            print(f"{epoch:>6} | {lt_m:>8.4f} | {la_m:>8.4f} | "
                  f"{val_mae:>8.4f} | {attn_acc:>9.3f} | "
                  f"[{mean_w[0]:.2f},{mean_w[1]:.2f}]")

        if val_mae < best_mae:
            best_mae = val_mae
            patience = 0
            model.save_weights('best.weights.h5')
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch}")
                break

    model.load_weights('best.weights.h5')
    print(f"\nBest val_mae: {best_mae:.4f}")
    print(f"Final attn_acc: {history['attn_acc'][-1]:.3f}  "
          f"(target > 0.85)")
    return history


def evaluate(model, Xq_val, Xw_val, Y_val, gs_val):
    pred, weights = model([Xq_val, Xw_val], return_weights=True)
    pred = pred.numpy()
    weights = weights.numpy()
    gs = gs_val.ravel()

    # denormalize
    td = np.array([30 * math.sqrt(1), 30 * math.sqrt(1), np.pi])
    pred_r = pred * td
    Y_r = Y_val * td

    err = np.abs(pred_r - Y_r)
    mae = (1.0 * err[:, 0].mean() + 1.0 * err[:, 1].mean() +
               20.0 * err[:, 2].mean())
    #mae = np.abs(pred_r - Y_r).mean(axis=0)
    rmse = np.sqrt(np.mean((pred_r - Y_r) ** 2, axis=0))

    pred_g = np.argmax(weights, axis=1)
    true_g = (gs > 0.5).astype(int)
    attn_acc = (pred_g == true_g).mean()

    active_w = np.where(gs < 0.5, weights[:, 0], weights[:, 1])
    inactive_w = np.where(gs < 0.5, weights[:, 1], weights[:, 0])
    std_w = weights.std(axis=0)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Weighted MAE  — vx:{mae:.4f}  ")
    print(f"RMSE — vx:{rmse[0]:.4f}  vy:{rmse[1]:.4f}  "
          f"omega:{rmse[2]:.4f}")
    print(f"\nAttn accuracy:    {attn_acc:.3f}  (target > 0.85)")
    print(f"Active weight:    {active_w.mean():.3f}  (target ~{1 - ALPHA})")
    print(f"Inactive weight:  {inactive_w.mean():.3f}  "
          f"(target ~{ALPHA})")
    print(f"Weight std:       [{std_w[0]:.3f}, {std_w[1]:.3f}]  "
          f"(> 0.15 = dynamic)")

    mask1 = gs < 0.5
    mask2 = gs > 0.5
    if mask1.sum() > 0:
        print(f"Goal1 phase:  w1={weights[mask1, 0].mean():.3f}  "
              f"w2={weights[mask1, 1].mean():.3f}")
    if mask2.sum() > 0:
        print(f"Goal2 phase:  w1={weights[mask2, 0].mean():.3f}  "
              f"w2={weights[mask2, 1].mean():.3f}")
    print("=" * 50)

    return {'mae': mae, 'rmse': rmse, 'attn_acc': attn_acc,
            'active_weight': active_w.mean()}

def plot_training(history):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(history['loss_track'])
    ax[0, 0].set_title('Tracking loss (weighted MAE)')
    ax[0, 0].set_xlabel('Epoch')

    ax[0, 1].plot(history['loss_attn'], color='coral')
    ax[0, 1].set_title('Attention supervision loss (KL)')
    ax[0, 1].set_xlabel('Epoch')

    ax[1, 0].plot(history['val_mae'], color='steelblue')
    ax[1, 0].set_title('Validation weighted MAE')
    ax[1, 0].set_xlabel('Epoch')

    ax[1, 1].plot(history['attn_acc'], color='green')
    ax[1, 1].axhline(0.85, color='gray', ls='--', label='Target 0.85')
    ax[1, 1].set_title('Attention goal accuracy')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: training_curves.pdf")


def plot_attention_trajectory(model, Xq_val, Xw_val,
                              gs_val, n=20):
    n = min(n, len(Xq_val))
    _, w = model([Xq_val[:n], Xw_val[:n]], return_weights=True)
    w = w.numpy()
    gs = gs_val[:n].ravel()
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].fill_between(t, 0, 1,
                         where=(gs < 0.5), alpha=0.1, color='steelblue')
    axes[0].fill_between(t, 0, 1,
                         where=(gs > 0.5), alpha=0.1, color='coral')
    axes[0].plot(t, w[:, 0], color='steelblue', lw=2,
                 label='Attention → goal1')
    axes[0].plot(t, w[:, 1], color='coral', lw=2,
                 label='Attention → goal2')
    axes[0].axhline(0.5, color='gray', lw=0.5, ls='--')
    axes[0].set_ylabel('Attention weight')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].set_title('Learned attention tracks active navigation goal')

    axes[1].step(t, gs, where='post', color='purple', lw=1.5)
    axes[1].set_ylabel('Active goal')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylim(-0.2, 1.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['goal1', 'goal2'])

    plt.tight_layout()
    plt.savefig('attention_trajectory.pdf', bbox_inches='tight',
                dpi=150)
    plt.close()
    print("Saved: attention_trajectory.pdf  (Figure 3)")


def plot_attention_heatmap(model, Xq_val, Xw_val, gs_val,
                           n=300):
    """Figure 4 — weight distribution and heatmap."""
    n = min(n, len(Xq_val))
    _, w = model([Xq_val[:n], Xw_val[:n]], return_weights=True)
    w = w.numpy()
    gs = gs_val[:n].ravel()

    active_w = np.where(gs < 0.5, w[:, 0], w[:, 1])
    inactive_w = np.where(gs < 0.5, w[:, 1], w[:, 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].imshow(w.T, aspect='auto', cmap='YlOrRd',
                        vmin=0, vmax=1)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['goal1', 'goal2'])
    axes[0].set_xlabel('Sample index')
    axes[0].set_title('Attention weights\n'
                      '(varied = learning; flat = not learning)')
    plt.colorbar(im, ax=axes[0])

    axes[1].hist(active_w, bins=30, alpha=0.7, color='teal',
                 label=f'Active (μ={active_w.mean():.2f})')
    axes[1].hist(inactive_w, bins=30, alpha=0.7, color='coral',
                 label=f'Inactive (μ={inactive_w.mean():.2f})')
    axes[1].axvline(ALPHA, color='gray', ls='--',
                    label=f'Target ({ALPHA})')
    axes[1].set_xlabel('Attention weight')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Active vs inactive goal weights')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('attention_heatmap.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: attention_heatmap.pdf     (Figure 4)")


def convert_to_tflite(model, Xq_tr, Xw_tr):
    """
    INT8 quantized TFLite with representative dataset calibration.
    Inputs:  query (1,8), waypoints (1,2,4)
    Output:  velocities (1,3)
    """
    print(f"\nConverting to TFLite...")

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 8], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 2, 4], dtype=tf.float32),
    ])
    def serving_fn(query, waypoints):
        out, _ = model([query, waypoints])
        return out

    n_cal = min(500, len(Xq_tr))
    cal_idx = np.random.choice(len(Xq_tr), n_cal, replace=False)

    def representative_dataset():
        for i in cal_idx:
            yield [Xq_tr[i:i + 1], Xw_tr[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"  Saved: {SAVE_PATH}  ({len(tflite_model) / 1024:.1f} KB)")

    # Verify quantization error
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    ind = interp.get_input_details()
    oud = interp.get_output_details()

    errors = []
    for i in cal_idx[:50]:
        tf_out, _ = model([Xq_tr[i:i + 1], Xw_tr[i:i + 1]])
        interp.set_tensor(ind[0]['index'], Xq_tr[i:i + 1])
        interp.set_tensor(ind[1]['index'], Xw_tr[i:i + 1])
        interp.invoke()
        tfl_out = interp.get_tensor(oud[0]['index'])
        errors.append(np.abs(tf_out.numpy() - tfl_out).mean())

    qerr = np.mean(errors)
    print(f"  Quantization MAE: {qerr:.5f}  "
          f"({'OK' if qerr < 0.01 else 'HIGH — recheck calibration'})")
    return tflite_model


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    print("IL-MPC Attention Policy")
    print("=" * 50)

    data = load_data(matlab_files)
    query, waypoints, past_u, goalswitch, target = extract_features(data)
    query_norm, wp_norm, past_norm, target_norm = normalize(query, waypoints, past_u, target)
    attn_targets = make_attention_targets(goalswitch)
    print("Python:", data[:5, 29])



    (Xq_tr, Xq_val, Xw_tr, Xw_val,
     Y_tr, Y_val, Yat_tr, gs_val) = random_split(
        query_norm, wp_norm, target_norm, attn_targets, goalswitch
    )
    #attn
    model = ILMPCAttentionPolicy(hidden=HIDDEN_NEURONS)
    _ = model([Xq_tr[:1], Xw_tr[:1]])  # force build
    model.summary()

    hist = train(model, Xq_tr, Xw_tr, Y_tr, Yat_tr,
                 Xq_val, Xw_val, Y_val, gs_val)

    metrics = evaluate(model, Xq_val, Xw_val, Y_val, gs_val)

    print(f"  Attention model     val_mae = {hist['val_mae'][-1]:.4f}")
    print(f"  Attention accuracy           = {metrics['attn_acc']:.3f}")

    #plot
    plot_training(hist)
    plot_attention_trajectory(model, Xq_val, Xw_val, gs_val)
    plot_attention_heatmap(model, Xq_val, Xw_val, gs_val)

    #convert
    #convert_to_tflite(model, Xq_tr, Xw_tr)

    print("\nFiles written:")
    #print(f"  {SAVE_PATH}            — deploy to Control Hub")
    print(f"  training_curves.pdf")
    print(f"  attention_trajectory.pdf  — Figure 3")
    print(f"  attention_heatmap.pdf     — Figure 4")

if __name__ == '__main__':
    main()
