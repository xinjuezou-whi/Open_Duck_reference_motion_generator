import matplotlib.pyplot as plt
# fit a polynomial per dimension of the data

import numpy as np
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# load the data
Y = np.array(json.load(open('ref_motion/0_-0.091_-0.121_-0.303.json'))["Frames"])

# Example: Generating synthetic time series data (500 samples, 59 dimensions)
np.random.seed(42)
X = np.linspace(0, 1, 500).reshape(-1, 1)  # Time as the feature
# Y = np.random.randn(500, 59)  # Simulated 59-dimensional data

degree = 3  # Change this for different polynomial degrees

# Store models
models = []

for dim in range(Y.shape[1]):  # Loop over each of the 59 dimensions
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)  # Transform time feature

    model = LinearRegression()
    model.fit(X_poly, Y[:, dim])  # Fit polynomial regression per dimension
    
    models.append((poly, model))  # Store transformer and model

# Example: Predicting for new time points
X_new = np.linspace(0, 1, 500).reshape(-1, 1)
X_new_poly = models[0][0].transform(X_new)  # Transform using the first stored poly
Y_pred = np.array([model.predict(X_new_poly) for poly, model in models]).T  # Predict for each dimension
for i in range(Y.shape[1]):
    # Plot example for one dimension
    plt.plot(X, Y[:, i], label="Original Data", alpha=0.5)
    plt.plot(X_new, Y_pred[:, i], label="Polynomial Fit", color='red')
    plt.legend()
    plt.show()
