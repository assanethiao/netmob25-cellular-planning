import matplotlib.pyplot as plt
import os

time = []
size = []
path = os.path.expanduser("~/ns-allinone-3.44/ns-3.44/DlRxPhyStats.txt")

with open(path, "r") as f:
    next(f)  
    for line in f:
        columns = line.split()
        time.append(float(columns[0]))
        size.append(float(columns[7]))  


bitrate = [(s * 8) / 1e6 for s in size]

plt.plot(time, bitrate)
plt.xlabel("Temps (s)")
plt.ylabel("Débit (Mbps)")
plt.title("Débit LTE en fonction du temps")
plt.grid()
plt.show()
