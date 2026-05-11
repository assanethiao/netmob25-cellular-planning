import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Charger les données
path = os.path.expanduser("~/ns-allinone-3.44/ns-3.44/trajectories_v02.csv")
df = pd.read_csv(path)

# Trier
df = df.sort_values(by="time")

# Liste des utilisateurs
nodes = df["node"].unique()

# Création figure
fig, ax = plt.subplots()

# Couleurs différentes
colors = ['r', 'b', 'g', 'orange', 'purple']

lines = {}
points = {}

# Initialisation des courbes
for i, node in enumerate(nodes):
    line, = ax.plot([], [], lw=2, label=f"Node {node}", color=colors[i % len(colors)])
    point, = ax.plot([], [], 'o', color=colors[i % len(colors)])
    lines[node] = line
    points[node] = point

# Limites
ax.set_xlim(df["x"].min(), df["x"].max())
ax.set_ylim(df["y"].min(), df["y"].max())

ax.set_title("Trajectoires des utilisateurs (Netmob25)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

# Initialisation
def init():
    for node in nodes:
        lines[node].set_data([], [])
        points[node].set_data([], [])
    return list(lines.values()) + list(points.values())

# Animation
def update(frame):
    current_time = sorted(df["time"].unique())[frame]

    for node in nodes:
        df_node = df[df["node"] == node]
        df_node = df_node[df_node["time"] <= current_time]

        x = df_node["x"].values
        y = df_node["y"].values

        lines[node].set_data(x, y)

        if len(x) > 0:
            points[node].set_data(x[-1], y[-1])

    return list(lines.values()) + list(points.values())

# Animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(df["time"].unique()),
    init_func=init,
    interval=100,
    blit=True
)

plt.show()
