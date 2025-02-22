import numpy as np
import json
import matplotlib.pyplot as plt
from glob import glob
import os

# ====== Load Data ======
all_files = glob("ref_motion/*.json")


def fit_ref_motion(file):
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    period = data["Placo"]["period"]
    fps = data["FPS"]
    frame_offsets = data["Frame_offset"][0]

    nb_steps_in_period = int(period * fps)
    _Y = Y_all[
        int(nb_steps_in_period) : int(nb_steps_in_period) + int(nb_steps_in_period)
    ]
    joints_pos = _Y[:, frame_offsets["joints_pos"]: frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"]: frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[:, frame_offsets["foot_contacts"]: frame_offsets["foot_contacts"]+2]
    base_linear_vel = _Y[:, frame_offsets["world_linear_vel"]: frame_offsets["world_angular_vel"]]
    base_angular_vel = _Y[:, frame_offsets["world_angular_vel"]: frame_offsets["joints_vel"]]

    Y = np.concatenate([joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel], axis=1)

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1)  # Time variable

    # Polynomial degree
    degree = 20

    # Store coefficients
    coefficients = {}

    # ====== Fit Polynomial Regression per Dimension ======
    for dim in range(Y.shape[1]):
        X_poly = np.vander(X.flatten(), degree + 1, increasing=True)
        coeffs, _, _, _ = np.linalg.lstsq(X_poly, Y[:, dim], rcond=None)
        coefficients[f"dim_{dim}"] = coeffs.tolist()

    return coefficients


all_coefficients = {}
for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"

    all_coefficients[name] = fit_ref_motion(file)


# Save coefficients
with open("polynomial_coefficients.json", "w") as f:
    json.dump(all_coefficients, f, indent=4)
exit()


# ====== Function to Sample at a Given Time and Dimension ======
def sample_polynomial(time: float, dimension: int, coefficients) -> float:
    """
    Evaluate the polynomial at a given time for a specific dimension.
    """
    if not (0 <= time <= 1):
        raise ValueError("Time must be between 0 and 1.")

    coeffs = coefficients.get(f"dim_{dimension}")
    if coeffs is None:
        raise ValueError(f"Dimension {dimension} not found in stored coefficients.")

    return sum(c * (time**i) for i, c in enumerate(coeffs))


# ====== Plot Fitting for Each Dimension ======
for dimension in range(Y.shape[1]):
    poly_samples = []
    ref_samples = []
    times = X.flatten()  # Use original time values for accurate mapping

    for i, t in enumerate(times):
        if i >= Y.shape[0]:
            break  # Prevent out-of-bounds errors

        poly_sample = sample_polynomial(t, dimension, coefficients)
        ref = Y[i, dimension]

        poly_samples.append(poly_sample)
        ref_samples.append(ref)

    # Plot original data vs. polynomial fit
    plt.figure(figsize=(6, 4))
    plt.plot(times[: len(ref_samples)], ref_samples, label="Original Data", alpha=0.5)
    plt.plot(
        times[: len(poly_samples)], poly_samples, label="Polynomial Fit", color="red"
    )
    plt.legend()
    plt.title(f"Dimension {dimension}")
    plt.xlabel("Time")
    plt.ylabel("Motion Value")
    plt.show()
