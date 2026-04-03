import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

matlab_files = ['v21cossinwithextrasauce.mat',
               'v22cossinwithextrasauce.mat']

ALPHA = 0.1 #this is just so softmax doesn't equal 1,0
VAL_PERCENT = 0.15 #15% validation data
HIDDEN_NEURONS = 64
EPOCHS        = 120
BATCH_SIZE    = 512
LR_INITIAL    = 1e-3
LR_DROP_EPOCH = 50       # halve LR here — matches your MATLAB schedule
LAMBDA_ATTN   = 0.1      # weight of attention supervision loss

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
    target = data[:, [29, 30, 31]]  # ideal vx, vy, omega


    query = np.concatenate([x_y, sin_cos_th, vel, goalswitch], axis=1)
    goal1 = np.concatenate([goal1_xy, sin_cos_g1], axis=1)
    goal2= np.concatenate([goal2_xy, sin_cos_g2], axis=1)
    waypoints = np.stack([goal1, goal2], axis=1)

    return  query, waypoints, past_u, goalswitch, target

def normalize(currentState, waypoints, past_u, target):
    state_norm = currentState / np.array([72, 72, 1, 1, 30, 30, np.pi, 1])
    wp_norm = waypoints / np.array([72, 72, 1, 1])
    past_norm = past_u / np.array([30, 30, np.pi])
    target_norm = target / np.array([30, 30, np.pi])

    return state_norm, wp_norm, past_norm, target_norm


def make_attention_targets(goalswitch):

    gs = goalswitch.ravel()
    targets = np.where(
        gs[:, None] == 0,
        [[ALPHA, 1 - ALPHA]],
        [[1 - ALPHA, ALPHA]]
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

        # Output head: 2*hidden -> 3
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
    td = np.array([30, 30, np.pi])
    pred_r = pred * td
    Y_r = Y_val * td

    mae = np.abs(pred_r - Y_r).mean(axis=0)
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
    print(f"MAE  — vx:{mae[0]:.4f}  vy:{mae[1]:.4f}  "
          f"omega:{mae[2]:.4f}")
    print(f"RMSE — vx:{rmse[0]:.4f}  vy:{rmse[1]:.4f}  "
          f"omega:{rmse[2]:.4f}")
    print(f"\nAttn accuracy:    {attn_acc:.3f}  (target > 0.85)")
    print(f"Active weight:    {active_w.mean():.3f}  (target ~{ALPHA})")
    print(f"Inactive weight:  {inactive_w.mean():.3f}  "
          f"(target ~{1 - ALPHA})")
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


data = load_data(matlab_files)
query, waypoints, past_u, goalswitch, target = extract_features(data)
query_norm, wp_norm, past_norm, target_norm = normalize(query, waypoints, past_u, goalswitch)
attention_targets = make_attention_targets(goalswitch)
(Xq_tr, Xq_val, Xw_tr, Xw_val,
     Y_tr, Y_val, Yat_tr, gs_val) = temporal_split(
        query_norm, wp_norm, target_norm, attention_targets, goalswitch
    )