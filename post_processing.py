import matplotlib.pyplot as plt
import numpy as np


for i in range(10):
    data = np.genfromtxt("waveform_{}".format(i), delimiter=" ")

    num_samples = (int)(len(data) / 3)
    sensor1 = data[:num_samples, 0]
    sensor2 = data[:num_samples, 1]

    t1 = np.linspace(0, num_samples * 2, num_samples)
    t2 = np.linspace(1, num_samples * 2 + 1, num_samples)

    fig = plt.figure(figsize=(13, 7.1))
    plt.plot(t1, sensor1)
    plt.plot(t2, sensor2)
    plt.plot((t1 + t2) / 2, sensor1 + sensor2)
    fig.tight_layout()
    plt.show()
