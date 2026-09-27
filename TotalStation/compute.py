import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import time

array = pd.read_csv("M_ARRAY.TXT", delimiter=",")
xyz_array = array[["East", "North", "Height"]].to_numpy()

# trasformazione per portare su piano xy
# 2. Vettori definiti dai punti MIC1, MIC2, MIC3
p1, p2, p3 = xyz_array[0], xyz_array[1], xyz_array[2]
v1 = p2 - p1
v2 = p3 - p1

# 3. Calcolo del vettore normale al piano
normal = np.cross(v1, v2)
normal_unit = normal / np.linalg.norm(normal)

# Target: allineare il vettore normale all'asse Z [0, 0, 1]
target_z = np.array([0.0, 0.0, 1.0])

# 4. Calcolo della matrice di rotazione (da normal_unit a [0, 0, 1])
rotation, _ = R.align_vectors([target_z], [normal_unit])

# 5. Applichiamo la trasformazione:
# a) Trasliamo mettendo MIC1 come origine (opzionale, assicura Z=0 esatto)
pts_centered = xyz_array - p1

# b) Applichiamo la rotazione a tutti i punti
pts_rotated = rotation.apply(pts_centered)

# rotazione attorno a z, per avere mic4 e mic5 paralleli a asse x
# vettore tra mic4 e mic5
v45 = xyz_array[3] - xyz_array[4]
v45_unit = v45 / np.linalg.norm(v45)
target_x = np.array([1.0, 0.0, 0.0])
z_rotation, _ = R.align_vectors([target_x], [v45_unit])
pts_final = z_rotation.apply(pts_rotated)

# Target: allineare il vettore v45 all'asse X [1, 0, 0]
target_x = np.array([1.0, 0.0, 0.0])

# Calcolo della matrice di rotazione (da v45_unit a [1, 0, 0])
rotation_z, _ = R.align_vectors([target_x], [v45_unit])

# Applichiamo la rotazione attorno a z
pts_rotated_z = rotation_z.apply(pts_rotated)

output_df = array.copy()
output_df["East"] = pts_final[:, 0]
output_df["North"] = pts_final[:, 1]
output_df["Height"] = pts_final[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# 3. Plot dei punti (Scatter Plot 3D)
scatter = ax.scatter(
    array["East"],
    array["North"],
    array["Height"],
    c="red",
    s=40,
    label="Measured points",
)
scatter_rotated = ax.scatter(
    output_df["East"],
    output_df["North"],
    output_df["Height"],
    c="blue",
    s=40,
    label="Rotated points",
)

# 4. Etichette degli assi
ax.set_aspect("equal")
ax.set_xlabel("Asse X")
ax.set_ylabel("Asse Y")
ax.set_zlabel("Asse Z")
ax.set_title("Scatter Plot 3D con Matplotlib")

ax.legend()

plt.show()

np.savetxt("triangle_measured.csv", pts_final, delimiter=",")
