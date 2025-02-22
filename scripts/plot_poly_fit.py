import os
import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt

polynomial_coefficients = json.load(open("polynomial_coefficients.json"))
all_files = glob("ref_motion/*.json")
tmp = json.load(open(all_files[0]))
period = tmp["Placo"]["period"]
fps = tmp["FPS"]
frame_offsets = tmp["Frame_offset"][0]
nb_steps_in_period = int(period * fps)


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


for file in all_files:
    name = os.path.basename(file).strip(".json")
    tmp = name.split("_")
    name = f"{tmp[1]}_{tmp[2]}_{tmp[3]}"
    banner = f"=== {name} ==="
    spacer = "=" * len(banner)
    print(spacer)
    print(banner)
    print(spacer)

    # Load reference motion data
    data = json.load(open(file))
    Y_all = np.array(data["Frames"])
    # Y = Y_all[
    #     int(nb_steps_in_period) : int(nb_steps_in_period) + int(nb_steps_in_period)
    # ]
    _Y = Y_all[
        int(nb_steps_in_period) : int(nb_steps_in_period) + int(nb_steps_in_period)
    ]
    joints_pos = _Y[:, frame_offsets["joints_pos"]: frame_offsets["left_toe_pos"]]
    joints_vel = _Y[:, frame_offsets["joints_vel"]: frame_offsets["left_toe_vel"]]
    foot_contacts = _Y[:, frame_offsets["foot_contacts"]: frame_offsets["foot_contacts"]+2]
    base_linear_vel = _Y[:, frame_offsets["world_linear_vel"]: frame_offsets["world_angular_vel"]]
    base_angular_vel = _Y[:, frame_offsets["world_angular_vel"]: frame_offsets["joints_vel"]]
    Y = np.concatenate([joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel], axis=1)

    nb_dim = Y.shape[1]

    # Generate time feature
    X = np.linspace(0, 1, Y.shape[0]).reshape(-1, 1)  # Time variable

    # Get coefficients for this motion
    coefficients = polynomial_coefficients[name]

    dimensions = []

    # ====== Plot Fitting for Each Dimension ======
    for dimension in range(nb_dim):
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

        dimensions.append((dimension, ref_samples, poly_samples))

    # ====== Plotting ======
    # make a grid figure with all the dimensions in the same plot

    nb_rows = int(np.sqrt(nb_dim))
    nb_cols = int(np.ceil(nb_dim / nb_rows))

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 8))
    axs = axs.flatten()

    for i, (dim, ref, poly) in enumerate(dimensions):
        ax = axs[i]
        ax.plot(ref, label="Original Data", alpha=0.5)
        ax.plot(poly, label="Polynomial Fit", color="red")
        # ax.set_title(f"Dimension {dim}")
        # ax.set_xlabel("Time")
        # ax.set_ylabel("Motion Value")
        ax.legend()

    plt.tight_layout()
    plt.show()
