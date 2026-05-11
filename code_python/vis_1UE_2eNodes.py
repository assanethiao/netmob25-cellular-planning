import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.expanduser("~/ns-allinone-3.44/ns-3.44/lte-data-avec_1UE_et_2eNodes.csv")
data = pd.read_csv(path)

# Position du UE
plt.figure()
plt.plot(data["X"], data["Y"])
plt.title("Trajectoire du UE")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

# Débit
plt.figure()
plt.plot(data["Time"], data["Throughput_Mbps"])
plt.title("Débit en fonction du temps")
plt.xlabel("Temps (s)")
plt.ylabel("Mbps")
plt.grid()
plt.show()
