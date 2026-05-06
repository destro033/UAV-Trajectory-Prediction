import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import MaxNLocator 


MODEL_A_PATH = "cmamba_results.npz"
MODEL_B_PATH = "patchtst_results.npz"

MODEL_A_NAME = "C-Mamba"
MODEL_B_NAME = "PatchTST"


# =========================
# Load saved results
# =========================
a = np.load(MODEL_A_PATH)
b = np.load(MODEL_B_PATH)


error_a = a["error_per_forecast_step"]
error_b = b["error_per_forecast_step"]

pred_a = a["preds"]
gt_a = a["gt"]

pred_b = b["preds"]
gt_b = b["gt"]

counts = a["flight_sample_counts"]
flight_ids = a["flight_ids"]

ade_a = [
    a["ade_first_step"],
    a["ade_96"],
    a["ade_last_step"]
]

ade_b = [
    b["ade_first_step"],
    b["ade_96"],
    b["ade_last_step"]
]

euclidean_a = a["euclidean_errors"]   
euclidean_b = b["euclidean_errors"]

mae_step_a = a["mae_per_forecast_step"]
mae_step_b = b["mae_per_forecast_step"]

mse_step_a = a["mse_per_forecast_step"]
mse_step_b = b["mse_per_forecast_step"]

steps = np.arange(1, len(mae_step_a) + 1)

plt.figure(figsize=(8, 4))
plt.plot(steps, mae_step_a, label=MODEL_A_NAME)
plt.plot(steps, mae_step_b, label=MODEL_B_NAME)

plt.xlabel("Forecast Step")
plt.ylabel("Combined XYZ MAE (m)")
plt.title("Combined XYZ MAE Across Forecast Horizon")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mae_forecast_step_curve.pdf", bbox_inches="tight")

steps = np.arange(1, len(mse_step_a) + 1)

plt.figure(figsize=(8, 4))
plt.plot(steps, mse_step_a, label=MODEL_A_NAME)
plt.plot(steps, mse_step_b, label=MODEL_B_NAME)

plt.xlabel("Forecast Step")
plt.ylabel("Combined XYZ MSE (m²)")
plt.title("Combined XYZ MSE Across Forecast Horizon")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mse_forecast_step_curve.pdf", bbox_inches="tight")

def plot_cdf_selected_steps(
    euclidean_a,
    euclidean_b,
    model_a_name,
    model_b_name,
    steps_to_plot=[0, 23, 47, 71, 95],
    save_path="cdf_selected_forecast_steps.pdf"
):
    plt.figure(figsize=(8, 5))

    for step in steps_to_plot:
        errors_a = np.sort(euclidean_a[:, step])
        cdf_a = np.arange(1, len(errors_a) + 1) / len(errors_a)

        errors_b = np.sort(euclidean_b[:, step])
        cdf_b = np.arange(1, len(errors_b) + 1) / len(errors_b)

        plt.plot(
            errors_a,
            cdf_a,
            linestyle="-",
            label=f"{model_a_name} Step {step + 1}"
        )

        plt.plot(
            errors_b,
            cdf_b,
            linestyle="--",
            label=f"{model_b_name} Step {step + 1}"
        )

    plt.xlabel("Euclidean Error (m)")
    plt.ylabel("CDF")
    plt.title("CDF of Euclidean Error at Selected Forecast Steps")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

plot_cdf_selected_steps(
    euclidean_a=euclidean_a,
    euclidean_b=euclidean_b,
    model_a_name=MODEL_A_NAME,
    model_b_name=MODEL_B_NAME,
    steps_to_plot=[0, 23, 47, 71, 95],
    save_path="cdf_selected_forecast_steps.pdf"
)


# Euclidean errors
labels = ["First", "Average (ADE)", "Final (FDE)"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 4))

bars_a = plt.bar(x - width / 2, ade_a, width, label=MODEL_A_NAME)
bars_b = plt.bar(x + width / 2, ade_b, width, label=MODEL_B_NAME)

def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

add_bar_labels(bars_a)
add_bar_labels(bars_b)

plt.xticks(x, labels)
plt.ylabel("Euclidean Error (m)")
plt.title("Error Comparison: First Step, ADE, and Final Step (FDE)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ade_fde_comparison.pdf", bbox_inches="tight")


#Euclidean error over forecast steps
steps = np.arange(1, len(error_a) + 1)

plt.figure(figsize=(8, 4))
plt.plot(steps, error_a, label=MODEL_A_NAME)
plt.plot(steps, error_b, label=MODEL_B_NAME)

plt.xlabel("Forecast Step")
plt.ylabel("Euclidean Error (m)")
plt.title("Euclidean Error Across Forecast Horizon")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("euclidean_error_curve.pdf", bbox_inches="tight")



# Helper: split flights
def split_by_flight(array, counts):
    flights = []
    start = 0

    for count in counts:
        end = start + count
        flights.append(array[start:end])
        start = end

    return flights


pred_a_flights = split_by_flight(pred_a, counts)
gt_a_flights = split_by_flight(gt_a, counts)

pred_b_flights = split_by_flight(pred_b, counts)
gt_b_flights = split_by_flight(gt_b, counts)



# 3D trajectory plots

def plot_3d_flight(
    gt,
    pred_model_a,
    pred_model_b,
    flight_id,
    forecast_step=0
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # shrink plot area to make room for Z label
    ax.set_position([0.05, 0.05, 0.75, 0.9])

    x_idx, y_idx, z_idx = 0, 1, 2

    ax.plot(
        gt[:, forecast_step, x_idx],
        gt[:, forecast_step, y_idx],
        gt[:, forecast_step, z_idx],
        label="Ground Truth",
        linewidth=2
    )

    ax.plot(
        pred_model_a[:, forecast_step, x_idx],
        pred_model_a[:, forecast_step, y_idx],
        pred_model_a[:, forecast_step, z_idx],
        label=MODEL_A_NAME
    )

    ax.plot(
        pred_model_b[:, forecast_step, x_idx],
        pred_model_b[:, forecast_step, y_idx],
        pred_model_b[:, forecast_step, z_idx],
        label=MODEL_B_NAME
    )

    # show only around 4 tick/grid lines per axis
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.ticklabel_format(axis="z", style="plain", useOffset=False)

    ax.set_xlabel("X(deg)", labelpad = 10)
    ax.set_ylabel("Y(deg)", labelpad =10)
    ax.set_zlabel("Z(m)", labelpad =10)
    ax.set_title(f"3D Trajectory Comparison")
    ax.legend()

    ax.set_box_aspect(None, zoom=0.85)
    
    plt.tight_layout()
    plt.savefig(f"trajectory_flight_{flight_id}.pdf", bbox_inches="tight")


for i, flight_id in enumerate(flight_ids):
    plot_3d_flight(
        gt=gt_a_flights[i],
        pred_model_a=pred_a_flights[i],
        pred_model_b=pred_b_flights[i],
        flight_id=flight_id,
        forecast_step=0
    )


plt.show()
