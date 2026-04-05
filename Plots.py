features_new = features = df[['position_x', 'position_y', 'position_z' ]]

feature_new_cols = features_new.columns

def evaluate_flight(
    flight_id,
    input_length=96,
    forecast_length=96
):
    """
    Returns:
        mae_xyz                -> (3,) MAE per axis (meters)
        ade_first              -> scalar
        ade_all                -> scalar
        error_per_timestep     -> (F,)
        cdf_distances          -> sorted distances (first forecast step)
        cdf_values             -> CDF values
        d80                    -> 80th percentile distance
        d90                    -> 90th percentile distance
    """

    # --------------------------------------------------
    # 1. Load + scale
    # --------------------------------------------------
    data_raw = df.loc[df["uid"] == flight_id, feature_new_cols].values
    data_scaled = scaler.transform(data_raw)

    dataset = TimeSeriesDataset(data_scaled, input_length, forecast_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    preds_all, targets_all = [], []

    # --------------------------------------------------
    # 2. Inference
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            preds_all.append(model(X).cpu())
            targets_all.append(y.cpu())

    y_pred = torch.cat(preds_all, dim=0)  # (N, V, F)
    y_true = torch.cat(targets_all, dim=0)

    N, V, F = y_pred.shape

    # --------------------------------------------------
    # 3. Inverse scaling
    # --------------------------------------------------
    y_pred_2d = y_pred.permute(0, 2, 1).reshape(-1, V)
    y_true_2d = y_true.permute(0, 2, 1).reshape(-1, V)

    y_pred_unscaled = scaler.inverse_transform(y_pred_2d)
    y_true_unscaled = scaler.inverse_transform(y_true_2d)

    y_pred_unscaled = y_pred_unscaled.reshape(N, F, V).transpose(0, 2, 1)
    y_true_unscaled = y_true_unscaled.reshape(N, F, V).transpose(0, 2, 1)

    # --------------------------------------------------
    # 4. Convert degrees → meters
    # --------------------------------------------------
    lat_deg = np.mean(y_true_unscaled[:, 1, :])
    lat_rad = np.deg2rad(lat_deg)

    disp = y_pred_unscaled - y_true_unscaled  # (N, V, F)

    disp[:, 0, :] *= 111320 * np.cos(lat_rad)  # longitude
    disp[:, 1, :] *= 111320                    # latitude
    # altitude already meters

    # --------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------

    # MAE per axis
    mae_xyz = np.abs(disp).mean(axis=(0, 2))

    # Euclidean per timestep
    l2_per_timestep = np.linalg.norm(disp, axis=1)  # (N, F)
    error_per_timestep = l2_per_timestep.mean(axis=0)

    ade_first = error_per_timestep[0]
    ade_all = error_per_timestep.mean()

    # --------------------------------------------------
    # 6. CDF (First Forecast Step)
    # --------------------------------------------------
    euclidean_first = l2_per_timestep[:, 0]  # (N,)

    sorted_dist = np.sort(euclidean_first)
    cdf_values = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)

    d80 = np.percentile(euclidean_first, 80)
    d90 = np.percentile(euclidean_first, 90)

    return (
        mae_xyz,
        ade_first,
        ade_all,
        error_per_timestep,
        sorted_dist,
        cdf_values,
        d80,
        d90,
        y_pred_unscaled,
        y_true_unscaled
    )

(mae_17,
 ade_first_17,
 ade_all_17,
 error_17,
 cdf_x_17,
 cdf_y_17,
 d80_17,
 d90_17,
 y_pred_unscaled,
 y_true_unscaled) = evaluate_flight(17)

pred17 = y_pred_unscaled
true17 = y_true_unscaled

(mae_18,
 ade_first_18,
 ade_all_18,
 error_18,
 cdf_x_18,
 cdf_y_18,
 d80_18,
 d90_18,
 y_pred_unscaled,
 y_true_unscaled) = evaluate_flight(18)

pred18 = y_pred_unscaled
true18 = y_true_unscaled

import numpy as np
import matplotlib.pyplot as plt

labels = ['X', 'Y', 'Z']

x = np.arange(len(labels))      # [0,1,2]
width = 0.35                    # bar width

fig, ax = plt.subplots(figsize=(9, 4))

bars1 = ax.bar(x - width/2, mae_17, width, label='Complicated Flight', color ='blue')
bars2 = ax.bar(x + width/2, mae_18, width, label='Simple Flight', color = 'red')

ax.set_ylabel('MAE (meters)')
#ax.set_title('MAE per Axis')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

max_val = max(np.max(mae_17), np.max(mae_18))
ax.set_ylim(0, max_val * 1.15)

# write values on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()


flights = ['Complicated Flight', 'Simple Flight']
ade_values = [ade_first_17, ade_first_18]

plt.figure(figsize=(6,5))
bars = plt.bar(flights, ade_values, color=['blue','red'])


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)

plt.ylabel('Euclidean distance (meters)')
#plt.title('Euclidean distance error per flight (Forecast Step = 0)')
plt.grid(axis='y', alpha=0.3)
plt.show()

# Compute errors for both flights
#error_17 = compute_euclidean_error_per_timestep(17)
#error_18 = compute_euclidean_error_per_timestep(18)

forecast_steps = np.arange(1, len(error_17)+1)

plt.figure(figsize=(9, 4))
plt.plot(forecast_steps, error_17, label='Complicated Flight', color='blue')
plt.plot(forecast_steps, error_18, label='Simple Flight', color='red')
plt.xlabel('Forecast Step')
plt.ylabel('Euclidean Error (meters)')
#plt.title('Euclidean Error vs Forecast Step')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

forecast_step = 0
x_idx, y_idx, z_idx = 0, 1, 2

# -----------------------------
# Flight 17
# -----------------------------
fig1 = plt.figure(figsize=(10,9))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot(true17[:, x_idx, forecast_step],
         true17[:, y_idx, forecast_step],
         true17[:, z_idx, forecast_step],
         label='Ground Truth', linewidth=2)

ax1.plot(pred17[:, x_idx, forecast_step],
         pred17[:, y_idx, forecast_step],
         pred17[:, z_idx, forecast_step],
         label='Prediction')

# ---- formatting fixes ----
ax1.ticklabel_format(style='plain', useOffset=False)

ax1.xaxis.set_major_locator(MaxNLocator(4))
ax1.yaxis.set_major_locator(MaxNLocator(4))
ax1.zaxis.set_major_locator(MaxNLocator(4))


ax1.set_xlabel('X (deg)', labelpad=10)
ax1.set_ylabel('Y (deg)', labelpad=10)
ax1.set_zlabel('Z (m)', labelpad=0.05)
#ax1.set_title('3D Trajectory Prediction (forecast step = 0)')
ax1.legend()


plt.subplots_adjust(right=0.85)

plt.tight_layout()
plt.show()


# -----------------------------
# Flight 18
# -----------------------------
fig2 = plt.figure(figsize=(10,9))
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot(true18[:, x_idx, forecast_step],
         true18[:, y_idx, forecast_step],
         true18[:, z_idx, forecast_step],
         label='Ground Truth', linewidth=2)

ax2.plot(pred18[:, x_idx, forecast_step],
         pred18[:, y_idx, forecast_step],
         pred18[:, z_idx, forecast_step],
         label='Prediction')

# ---- formatting fixes ----
ax2.ticklabel_format(style='plain', useOffset=False)

ax2.xaxis.set_major_locator(MaxNLocator(4))
ax2.yaxis.set_major_locator(MaxNLocator(4))
ax2.zaxis.set_major_locator(MaxNLocator(4))


ax2.set_xlabel('X (deg)', labelpad=10)
ax2.set_ylabel('Y (deg)', labelpad=10)
ax2.set_zlabel('Z (m)', labelpad=0.05)
#ax2.set_title('3D Trajectory Prediction (forecast step = 0)')
ax2.legend()


plt.subplots_adjust(right=0.85)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))

plt.plot(cdf_x_17, cdf_y_17, label='Complicated Flight', linewidth=2)
plt.plot(cdf_x_18, cdf_y_18, label='Simple Flight', linewidth=2, color = 'red')

plt.xlabel('Euclidean distance (meters)')
plt.ylabel('CDF')
#plt.title('CDF of Euclidean distance error (First Forecast Step)')

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.xlim(0, 15)
plt.xticks(np.arange(0, 16, 1))

plt.show()
