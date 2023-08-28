import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("DensityWaterEthanol.pkl", "rb") as f:
    data = pickle.load(f)

    temp = data[:, 0]
    mol_frac_ethanol = data[:, 1]
    density = data[:, 2]

    X = np.array([temp, mol_frac_ethanol, np.ones(len(temp))]).T

    XT_X_inv = np.linalg.inv(X.T @ X)
    p = XT_X_inv @ X.T @ density

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})
    ax.scatter(temp, mol_frac_ethanol, density)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mol fraction ethanol")
    ax.set_zlabel("Density")
    plt.show()
