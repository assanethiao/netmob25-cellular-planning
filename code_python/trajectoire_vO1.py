import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Charger les données
path = os.path.expanduser("~/ns-allinone-3.44/ns-3.44/trajectories.csv")
df = pd.read_csv(path)

# Trier par temps (important)
df = df.sort_values(by="time")

# Extraire
x = df["x"].values
y = df["y"].values

# Création figure
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
point, = ax.plot([], [], 'ro')  # point mobile

# Limites auto
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))

ax.set_title("Trajectoire utilisateur")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Initialisation
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# Animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    point.set_data(x[frame], y[frame])
    return line, point

# Animation fluide
ani = FuncAnimation(
    fig,
    update,
    frames=len(x),
    init_func=init,
    interval=100,   # vitesse (ms)
    blit=True
)

plt.show()
