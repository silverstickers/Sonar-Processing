from multiprocessing import Process, Queue
import serial
import time
import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import optimize as opt
import numpy as np
from math import pi
import itertools
from functools import partial

WRITE = False
offsets = offsets = [0, -4, -4.5, -4.5]
amplitude_weights = [1, 1, 1, 1]


f = 40000
c = 340
wavelength = c/f
k = 2*pi/wavelength
pi_over_180 = pi / 180

spacing = 0.0105
spacing_over_c = spacing / c



def convertToInt(array):
    # return [int.from_bytes(array[2*i:2*i+1], byteorder='little') \
            # for i in range((int)(bytes_per_array/2))]
    new_array = []
    for a in zip(array[::2], array[1::2]):
        new_array.append(int.from_bytes(a, byteorder='little'))
    return new_array


def max_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)


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
    sensor = y
    fit_window = window_size
    fit_sensor = lambda x: 0
    r = 0
    par, _ = opt.curve_fit(
        modulated_sine,
        t1[0:2*fit_window] - 0.5*fit_window*4,
        sensor[0:2*fit_window],
        [40000*1e-6, 0, 11, 0.1, 0.1, 1700],
        maxfev = 1000
    )
    temp = lambda x, pars=par, r=r, fit_window=fit_window: ((r)*fit_window*4 < x) * (x < (r+1)*fit_window*4) * modulated_sine(x-(r+0.5)*fit_window*4, pars)
    temp2 = lambda x, f=fit_sensor, t=temp: f(x) + t(x)
    fit_sensor = temp2

    for r in range(1, int(len(x)/fit_window)-1):
        try:
            par, _ = opt.curve_fit(
                modulated_sine,
                t1[int((r-0.7)*fit_window):int((r+1.7)*fit_window)] - (r+0.5)*fit_window*4,
                sensor[int((r-0.7)*fit_window):int((r+1.7)*fit_window)],
                [40000*1e-6, 0.1, 1, 0.1, 0.01, 1700],
                maxfev=1000
            )
        except RuntimeError:
            par = [0, 0, 0, 0, 0, 0]
        if r == int(1600/fit_window):
            print(par[0])
        temp = lambda x, pars=par, r=r, fit_window=fit_window: ((r)*fit_window*4 < x) * (x < (r+1)*fit_window*4) * modulated_sine(x-(r+0.5)*fit_window*4, pars)
        temp2 = lambda x, f=fit_sensor, t=temp: f(x) + t(x)
        fit_sensor = temp2
    return fit_sensor







def plot_2d(waveforms, delta_theta=60):
    # sin(m_angle*pi/180.)*k*m_spacing
    NUM_RECEIVERS = len(waveforms)
    num_samples = len(waveforms[0])

    times = [np.linspace(i, num_samples*4 + i, num_samples) for i in range(NUM_RECEIVERS)]

    fit_window = 23

    start_time = timeit.default_timer()
    
    fit_sensors = []

    for t, w in zip(times, waveforms):
        fit_sensors.append(fit_data(t[50:-500], w[50:-500], fit_window))

    

    elapsed = timeit.default_timer() - start_time
    print("########\nFits took {} ms".format(elapsed*1e3))
    start_time = timeit.default_timer()

    theta = np.arange(-delta_theta/2, delta_theta/2, 0.8)
    rho   = np.arange(0.23, 0.37, 0.001)

    timeDiff = np.sin(theta*pi_over_180)*spacing_over_c

    window_size = 17 # sollte ungerade sein
    rho_plot = rho[int(window_size/2):-int(window_size/2)]
    timeDistance = 2e6 * rho / 340

    z_plot = []
    for t_ in timeDiff:
        # timeDiff = np.sin(t_*pi_over_180)*spacing_over_c
        values = np.zeros(len(timeDistance))
        for i in range(NUM_RECEIVERS):
            values += amplitude_weights[i]*fit_sensors[i](offsets[i]+timeDistance-i*1e6*t_)
        
        values = np.abs(values)
        z_plot.append(max_rolling1(values, window_size))
        
    z_plot = np.array(z_plot)
    z_min = z_plot.min()
    z_max = z_plot.max()
    z_plot -= z_min
    z_plot /= z_max
    z_plot = np.exp(z_plot)

    elapsed = timeit.default_timer() - start_time
    print("Digital scan took {} ms".format(elapsed*1e3))

    return theta, rho_plot, z_plot



def handle_arduino_output(queue, delta_theta=60):
    with serial.Serial(port='COM4', baudrate=115200, timeout=5.0) as arduino:
        filesWritten = 0
        while True:

            # align to starting pattern
            count = 0
            pattern = [1, 200, 31, 41]
            patternPosition = 0
            while True:
                value = arduino.read(1)[0]
                if value == pattern[patternPosition]:
                    count += 1
                else:
                    count = 0
                    patternPosition = 0
                if count > 24:
                    # print("Found {} {} in a row".format(count, pattern[patternPosition]))
                    patternPosition += 1
                    count = 0
                    if patternPosition == len(pattern):
                        break
            
            try:
                message = arduino.readline()
                print(message.decode("utf-8"))
                
                bytes_per_array = int.from_bytes(arduino.read(4), byteorder='little')
                num_arrays      = int.from_bytes(arduino.read(4), byteorder='little')
                print("Bytes per array: {}".format(bytes_per_array))
                start_time = timeit.default_timer()
                arrays = []
                for i in range(num_arrays):
                    arrays.append(arduino.read(bytes_per_array))
                elapsed = timeit.default_timer() - start_time

                
                print("Timed the serial read. Elapsed time: {:2.3f} ms".format(elapsed * 1000))
                
                
                start_time = time.perf_counter_ns()
                for i in range(num_arrays):
                    arrays[i] = convertToInt(arrays[i])
                
                elapsed = time.perf_counter_ns() - start_time

                if filesWritten < 5 and WRITE:
                    with open('waveforms/0deg_waveform_{}'.format(filesWritten), 'w') as f:
                        for i in range(len(arrays[0])):
                            for j in range(len(arrays)):
                                f.write("{} ".format(arrays[j][i]))
                            f.write("\n")
                    filesWritten += 1
                
                
                theta, rho_plot, z_plot = plot_2d(arrays, delta_theta)
                # queue.put((arrays[0][:-1000:2], theta, rho_plot, z_plot,))
                queue.put((0, theta, rho_plot, z_plot,))
                # queue.put((arrays[0][:-1500], 0, 0, 0,))

                message = arduino.readline()
                print(message.decode("utf-8"))
                print("\n")

            except UnicodeDecodeError:
                pass


def make_frame(delta_theta, queue, ax1, ax2):
    array, theta, rho_plot, z_plot = queue.get(timeout=5.0)

    
    colormap = plt.get_cmap('inferno')
    z_min = z_plot.min()
    z_max = z_plot.max()
    norm = mpl.colors.Normalize(z_min, z_max)
    ax1.clear()
    ax1.set_thetamin(-delta_theta/2)
    ax1.set_thetamax(delta_theta/2)
    ax1.set_rmin(0.2)
    ax1.set_theta_offset(pi/2)

    for t, values in zip(theta, z_plot):
        ax1.scatter([-t*pi/180]*len(rho_plot), rho_plot, c=values, s=6, cmap=colormap, norm=norm, linewidths=0)
    
    # xarr = range(len(array))
    # ax2.clear()
    # ax2.plot(xarr, array)




def main():
    delta_theta = 60

    queue = Queue()
    arduino_process = Process(target=handle_arduino_output, args=(queue,delta_theta,))
    arduino_process.daemon = True
    arduino_process.start()

    fig = plt.figure()
    ax1 = plt.subplot(1, 1, 1, polar=True)
    # ax2 = plt.subplot(1, 2, 2)
    next_frame = partial(make_frame, queue=queue, ax1=ax1, ax2=None)

    ani = animation.FuncAnimation(fig, next_frame, itertools.repeat(60), interval=10, save_count=1)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
