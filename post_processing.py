from cmath import phase
from stat import FILE_ATTRIBUTE_SYSTEM
import matplotlib.pyplot as plt
import numpy as np
from math import sin, pi, asin
import scipy.optimize as opt
import timeit

def modulated_sine(x, *pars):
    try:
        if len(pars) == 1:
            pars = pars[0]
        f, phi, p0, p1, p2, offset = pars
    except ValueError:
        print(pars)
        raise
    local_aplitude = (p0 + p1*x + p2*x**2)
    return local_aplitude*np.sin(2*pi*f*x + phi) + offset


def fit_data(x, y, window_size):
    t1 = x
    sensor1 = y
    fit_window = window_size
    fit_sensor1 = lambda x: 0
    r = 0
    par, _ = opt.curve_fit(
        modulated_sine,
        t1[0:2*fit_window] - 0.5*fit_window*4,
        sensor1[0:2*fit_window],
        [40000*1e-6, 0, 11, 0.1, 0.1, 1700],
        maxfev = 5000
    )
    temp = lambda x, pars=par, r=r, fit_window=fit_window: ((r)*fit_window*4 < x) * (x < (r+1)*fit_window*4) * modulated_sine(x-(r+0.5)*fit_window*4, pars)
    temp2 = lambda x, f=fit_sensor1, t=temp: f(x) + t(x)
    fit_sensor1 = temp2

    for r in range(1, int(num_samples/fit_window)-1):
        try:
            par, _ = opt.curve_fit(
                modulated_sine,
                t1[int((r-0.7)*fit_window):int((r+1.7)*fit_window)] - (r+0.5)*fit_window*4,
                sensor1[int((r-0.7)*fit_window):int((r+1.7)*fit_window)],
                [40000*1e-6, 0.1, 1, 0.1, 0.01, 1700],
                maxfev=2000
            )
        except RuntimeError:
            par = [0, 0, 0, 0, 0, 0]
        # print(par)
        temp = lambda x, pars=par, r=r, fit_window=fit_window: ((r)*fit_window*4 < x) * (x < (r+1)*fit_window*4) * modulated_sine(x-(r+0.5)*fit_window*4, pars)
        temp2 = lambda x, f=fit_sensor1, t=temp: f(x) + t(x)
        fit_sensor1 = temp2
    return fit_sensor1

if __name__ == "__main__":
    f = 40000
    c = 340
    wavelength = c/f
    k = 2*pi/wavelength

    spacing = 0.0105
    # sin(m_angle*pi/180.)*k*m_spacing

    for i in range(1,2):
        data = np.genfromtxt("waveforms/0deg_waveform_{}".format(i), delimiter=" ")

        num_samples = (int)(len(data))
        sensor1 = data[:num_samples, 0]
        sensor2 = data[:num_samples, 1]
        sensor3 = data[:num_samples, 2]
        sensor4 = data[:num_samples, 3]

        t1 = np.linspace(0, num_samples * 4, num_samples)
        t2 = np.linspace(1, num_samples * 4 + 1, num_samples)
        t3 = np.linspace(2, num_samples * 4 + 2, num_samples)
        t4 = np.linspace(3, num_samples * 4 + 3, num_samples)

        fit_sensor1 = lambda x: 0
        fit_window = 23

        start_time = timeit.default_timer()
        
        fit_sensor1 = fit_data(t1, sensor1, fit_window)
        fit_sensor2 = fit_data(t2, sensor2, fit_window)
        fit_sensor3 = fit_data(t3, sensor3, fit_window)
        fit_sensor4 = fit_data(t4, sensor4, fit_window)
        

        elapsed = timeit.default_timer() - start_time
        print("Fits took {} ms".format(elapsed*1e3))
        print(sum( (sensor1 - fit_sensor1(t1))**2 ) / num_samples)

        t_smoothed = np.linspace(1, num_samples*4-400, 10*num_samples)

        fig = plt.figure(figsize=(13, 7.1))
        # plt.plot(t1, sensor1, label="data")
        plt.plot(t_smoothed, fit_sensor1(t_smoothed), label="1")
        # plt.plot(t2, sensor2 + 1000)
        plt.plot(t_smoothed, fit_sensor2(t_smoothed), label="2")
        # plt.plot(t3, sensor3 + 2000)
        plt.plot(t_smoothed, fit_sensor3(t_smoothed), label="3")
        # plt.plot(t4, sensor4 + 3000)
        plt.plot(t_smoothed, fit_sensor4(t_smoothed), label="4")

        plt.legend()
        # plt.plot((t1 + t2) / 2, sensor1 + sensor2)

        # for timeDiff_micros in range(-64, 65, 16):
        #     angle = asin((timeDiff_micros+1)*1e-6*f/(k*spacing)) * 180 / pi
        #     print(angle)
        #     plt.plot(t1, 6000 + angle*700 + sensor1 + \
        #                 np.roll(sensor2, (int)(timeDiff_micros/4)) + \
        #                 np.roll(sensor3, 2*(int)(timeDiff_micros/4)) + \
        #                 np.roll(sensor4, 3*(int)(timeDiff_micros/4)))

        # for angle in range(-15, 20, 5):
        #     phaseDiff = sin(angle*pi/180.)*k*spacing
        #     timeDiff = phaseDiff/f
        #     print(timeDiff*1e6)
        #     plt.plot(t1, 6000 + angle*400 + sensor1 + \
        #                 np.roll(sensor2, -(int)(1e6*timeDiff/4)) + \
        #                 np.roll(sensor3, -2*(int)(1e6*timeDiff/4)) + \
        #                 np.roll(sensor4, -3*(int)(1e6*timeDiff/4)))

        fig.tight_layout()
        plt.show()

        
