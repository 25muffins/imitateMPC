import math

import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

matlab_files = ['GRUDiffV1.mat',
                'GRUDiffV2.mat']

VAL_PERCENT = 0.15 #15% validation data
EPOCHS        = 120
BATCH_SIZE    = 512
LR_INITIAL    = 0.001
LR_DROP_EPOCH = 50       # same as MATLAB
SAVE_PATH = 'tfliteFiles/Test0_Baseline.tflite'

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
                 goalswitch,
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
    gs_val           = goalswitch[val_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  "
          f"Test: {n_test}")
    return (Xq_tr, Xq_val, Xw_tr, Xw_val,
            Y_tr, Y_val, gs_val)


def convert_to_tflite(model, Xq_tr, Xw_tr):
    """
    INT8 quantized TFLite with representative dataset calibration.
    Inputs:  query (1,8), waypoints (1,2,4)
    Output:  velocities (1,3)
    """

    _ = model([Xq_tr[:1], Xw_tr[:1]])

    print(f"\nConverting to TFLite...")
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 8], dtype=tf.float32),  # query
        tf.TensorSpec(shape=[1, 2, 4], dtype=tf.float32),  # waypoints
    ])

    def serving_fn(query, waypoints):
        out = model([query, waypoints])
        return out

    cf = serving_fn.get_concrete_function()

    cal_idx = np.random.choice(len(Xq_tr), min(500, len(Xq_tr)), replace=False)

    def representative_dataset():
        for i in cal_idx:
            yield [Xq_tr[i:i + 1], Xw_tr[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [cf], model
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
        tf_out = model([Xq_tr[i:i + 1], Xw_tr[i:i + 1]])
        interp.set_tensor(ind[0]['index'], Xq_tr[i:i + 1])
        interp.set_tensor(ind[1]['index'], Xw_tr[i:i + 1])
        interp.invoke()
        tfl_out = interp.get_tensor(oud[0]['index'])
        errors.append(np.abs(tf_out.numpy() - tfl_out).mean())

    qerr = np.mean(errors)
    print(f"  Quantization MAE: {qerr:.5f}  "
          f"({'OK' if qerr < 0.01 else 'HIGH — recheck calibration'})")
    return tflite_model


class MLPBaseline(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # current (4) + g1 (4) + g2 (4) + gs = 13 input features
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(450, activation='relu',
                                  input_shape=(13,)),
            #tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(45, activation='relu'),
            tf.keras.layers.Dense(3),
            tf.keras.layers.Activation('tanh')
        ])

    def call(self, inputs):
        query, waypoints = inputs
        wp_flat = tf.reshape(waypoints, [-1, 8])
        inputFeatures = tf.concat([tf.gather(query, [0,1,2,3,7], axis = 1), wp_flat], axis = -1)
        return self.net(inputFeatures)
        #return self.net(tf.concat([query, wp_flat], axis=-1))


def train_baseline(Xq_tr, Xw_tr, Y_tr, Xq_val, Xw_val, Y_val):
    print("\nTraining MLP baseline for ablation comparison...")
    baseline = MLPBaseline()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        epsilon=1e-8,
        weight_decay=1e-4       # L2Regularization
    )
    @tf.function
    def step(Xq, Xw, Y):
        with tf.GradientTape() as tape:
            p = baseline([Xq, Xw])
            err = tf.abs(p - Y)
            loss = (1.0 * tf.reduce_mean(err[:, 0]) +
                    1.0 * tf.reduce_mean(err[:, 1]) +
                    1.0 * tf.reduce_mean(err[:, 2]))/3  #mean
        g = tape.gradient(loss, baseline.trainable_variables)
        optimizer.apply_gradients(zip(g, baseline.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        #this is so that we can print out our loss
        lossList = []

        if epoch == LR_DROP_EPOCH:
            optimizer.learning_rate.assign(LR_INITIAL * 0.5)
        idx = np.random.permutation(len(Xq_tr))
        for i in range(0, len(idx), BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            lossList.append(step(Xq_tr[b], Xw_tr[b], Y_tr[b]))
        if epoch % 10 == 0:
            print(f"  Epoch: {epoch}/{EPOCHS}  |  LOSS: {np.mean(lossList)}")




    vp = baseline([Xq_val, Xw_val]).numpy()
    baseline.summary()
    err = np.abs(vp - Y_val)
    #mae = (1.0 * err[:, 0].mean() + 1.0 * err[:, 1].mean() +
    #       1.0 * err[:, 2].mean())
    mae = err.mean()
    mse = np.mean((vp - Y_val) ** 2)
    print(f"  val_mae: {mae:.4f}  |  val_mse: {mse:.4f}")
    return baseline, mae, mse


def main():
    tf.random.set_seed(41)
    np.random.seed(41)

    data = load_data(matlab_files)
    query, waypoints, past_u, goalswitch, target = extract_features(data)
    query_norm, wp_norm, past_norm, target_norm = normalize(query, waypoints, past_u, target)

    (Xq_tr, Xq_val, Xw_tr, Xw_val,
     Y_tr, Y_val, gs_val) = random_split(
        query_norm, wp_norm, target_norm, goalswitch
    )

    baseline, base_mae, base_mse = train_baseline(
       Xq_tr, Xw_tr, Y_tr, Xq_val, Xw_val, Y_val
    )

    print(f"\nAblation table:")
    print(f"  MLP baseline        val_mae = {base_mae:.4f}  |  val_mse = {base_mse:.4f}")

    convert_to_tflite(baseline, Xq_tr, Xw_tr)

if __name__ == '__main__':
    main()
