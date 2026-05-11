import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# === Charger les données ===
csv_path = os.path.expanduser("~/ns-allinone-3.44/ns-3.44/trajectories_v03_50UE_5eNode.csv")
df = pd.read_csv(csv_path)

# Trier par temps
df = df.sort_values(by="time")

# Liste des noeuds
nodes = df["node"].unique()

# === Charger image de fond ===
img_path = os.path.expanduser("~/ns-allinone-3.44/code_python/map1.png")  # ton image ici
img = plt.imread(img_path)

# === Création figure ===
fig, ax = plt.subplots(figsize=(8, 8))

# Définir limites
xmin, xmax = df["x"].min(), df["x"].max()
ymin, ymax = df["y"].min(), df["y"].max()

# Afficher image en background
ax.imshow(img, extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=0.5)

# Couleurs
colors = plt.cm.tab20.colors  # palette auto

lines = {}
points = {}

# Initialisation des courbes
for i, node in enumerate(nodes):
    color = colors[i % len(colors)]
    
    line, = ax.plot([], [], lw=1.5, color=color)
    point, = ax.plot([], [], 'o', color=color, markersize=4)
    
    lines[node] = line
    points[node] = point

# Labels
ax.set_title("Trajectoires des utilisateurs (Netmob25)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# === Initialisation ===
def init():
    for node in nodes:
        lines[node].set_data([], [])
        points[node].set_data([], [])
    return list(lines.values()) + list(points.values())

# === Animation ===
times = sorted(df["time"].unique())

def update(frame):
    current_time = times[frame]

    for node in nodes:
        df_node = df[(df["node"] == node) & (df["time"] <= current_time)]

        x = df_node["x"].values
        y = df_node["y"].values

        lines[node].set_data(x, y)

        if len(x) > 0:
            points[node].set_data(x[-1], y[-1])

    return list(lines.values()) + list(points.values())

# === Lancer animation ===
ani = FuncAnimation(
    fig,
    update,
    frames=len(times),
    init_func=init,
    interval=200,
    blit=True
)

plt.show()
