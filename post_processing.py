import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import sin, pi, asin
import scipy.optimize as opt
import timeit

NUM_RECEIVERS = 4

def max_rolling1(a, window,axis =1):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.max(rolling,axis=axis)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def hanning_weights(n):
    return 0.54 - 0.46*np.cos(2*pi*n/(NUM_RECEIVERS-1))

def modulated_sine(x, *pars):
    try:
        if len(pars) == 1:
            pars = pars[0]
        f, phi, p0, p1, p2, offset = pars
    except ValueError:
        print(pars)
        raise
    return (p0 + p1*x + p2*x**2) * np.sin(2*pi*f*x + phi) + offset


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
        if r == int(1600/fit_window):
            print(par[0])
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

    i = 2

    data = np.genfromtxt("waveforms/0deg_waveform_{}".format(i), delimiter=" ")

    num_samples = (int)(len(data))-1000
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

    t_smoothed = np.linspace(1, num_samples*4-400, 3*num_samples)

    fig = plt.figure(figsize=(13, 7.1))
    # plt.plot(t1, sensor1, label="data")
    # plt.plot(t_smoothed, fit_sensor1(t_smoothed), label="1")
    # plt.plot(t2, sensor2)
    # plt.plot(t_smoothed, fit_sensor2(t_smoothed), label="2")
    # plt.plot(t3, sensor3 + 2000)
    # plt.plot(t_smoothed, fit_sensor3(t_smoothed), label="3")
    # plt.plot(t4, sensor4 + 3000)
    # plt.plot(t_smoothed, fit_sensor4(t_smoothed), label="4")

    
    # plt.plot(t_smoothed, fit_sensor1(t_smoothed) + fit_sensor2(t_smoothed))
    offsets = [0, -4, -4.5, -4.5]
    amplitude_weights = [1, 1, 1, 1] #[hanning_weights(i) for i in range(NUM_RECEIVERS)]
    # for timeDiff_micros in range(-15, 16, 1):
    #     angle = asin((timeDiff_micros)*1e-6*2*pi*f/(k*spacing)) * 180 / pi
    #     print(angle)
    #     if timeDiff_micros==0:
    #         plt.plot(t_smoothed, 6000 + timeDiff_micros*1500 +\
    #             fit_sensor1(offsets[0]+t_smoothed) +\
    #             fit_sensor2(offsets[1]+t_smoothed - 1*timeDiff_micros) +\
    #             fit_sensor3(offsets[2]+t_smoothed - 2*timeDiff_micros) +\
    #             fit_sensor4(offsets[3]+5+t_smoothed - 3*timeDiff_micros), 'k')
    #     else:
    #         plt.plot(t_smoothed, 6000 + timeDiff_micros*1500 +\
    #             fit_sensor1(offsets[0]+t_smoothed) +\
    #             fit_sensor2(offsets[1]+t_smoothed - 1*timeDiff_micros) +\
    #             fit_sensor3(offsets[2]+t_smoothed - 2*timeDiff_micros) +\
    #             fit_sensor4(offsets[3]+5+t_smoothed - 3*timeDiff_micros))
        # plt.plot(t1, 6000 + angle*700 + sensor1 + \
        #             np.roll(sensor2, (int)(timeDiff_micros/4)) + \
        #             np.roll(sensor3, 2*(int)(timeDiff_micros/4)) + \
        #             np.roll(sensor4, 3*(int)(timeDiff_micros/4)))


    for angle in range(-26, 30, 2):
        phaseDiff = sin(angle*pi/180.)*k*spacing
        timeDiff = phaseDiff/(2*pi*f)
        print("angle: {:<3.1f} \t phase diff: {:<3.2f} deg \t time diff: {:<3.2f} µs"\
                .format(angle, phaseDiff*180/pi, timeDiff*1e6)
            )
        # plt.figure(1)
        if angle == 0:
            plt.plot(340*100*t_smoothed*1e-6/2, 6000 + angle*1000 +\
            1*amplitude_weights[0]*fit_sensor1(offsets[0]+t_smoothed) +\
            1*amplitude_weights[1]*fit_sensor2(offsets[1]+t_smoothed - 1e6*timeDiff) +\
            1*amplitude_weights[2]*fit_sensor3(offsets[2]+t_smoothed - 2e6*timeDiff) +\
            1*amplitude_weights[3]*fit_sensor4(offsets[3]+t_smoothed - 3e6*timeDiff), 'k')
        else:
            plt.plot(340*100*t_smoothed*1e-6/2, 6000 + angle*1000 +\
                1*amplitude_weights[0]*fit_sensor1(offsets[0]+t_smoothed) +\
                1*amplitude_weights[1]*fit_sensor2(offsets[1]+t_smoothed - 1e6*timeDiff) +\
                1*amplitude_weights[2]*fit_sensor3(offsets[2]+t_smoothed - 2e6*timeDiff) +\
                1*amplitude_weights[3]*fit_sensor4(offsets[3]+t_smoothed - 3e6*timeDiff))





    f3 = plt.figure(3)
    ax = plt.subplot(1, 1, 1, polar=True)
    ax.set_thetamin(-40)
    ax.set_thetamax(40)
    ax.set_rmin(0.2)
    ax.set_theta_offset(pi/2)

    theta = np.arange(-30, 30, 0.5)
    rho   = np.arange(0.22, 0.4, 0.0004)

    window_size = 37 # sollte ungerade sein
    rho_plot = rho[int(window_size/2):-int(window_size/2)]
    timeDistance = 2e6 * rho / 340

    
    z_plot = []
    for t in theta:
        phaseDiff = sin(t*pi/180)*k*spacing
        timeDiff = phaseDiff/(2*pi*f)

        values = np.abs(
            amplitude_weights[0]*fit_sensor1(offsets[0]+timeDistance) +\
            amplitude_weights[1]*fit_sensor2(offsets[1]+timeDistance-1e6*timeDiff) +\
            amplitude_weights[2]*fit_sensor3(offsets[2]+timeDistance-2e6*timeDiff) +\
            amplitude_weights[3]*fit_sensor4(offsets[3]+timeDistance-3e6*timeDiff)
        )
        values = max_rolling1(values, window_size)
        # values = moving_average(values, window_size)
        z_plot.append(values)

    
    z_plot = np.array(z_plot)
    z_min = z_plot.min()
    z_max = z_plot.max()
    z_plot -= z_min
    z_plot /= z_max
    z_plot = np.exp(z_plot)
    z_min = z_plot.min()
    z_max = z_plot.max()

    colormap = plt.get_cmap('inferno')
    norm = mpl.colors.Normalize(z_min, z_max)
    for t, values in zip(theta, z_plot):
        ax.scatter([t*pi/180]*len(rho_plot), rho_plot, c=values, s=5, cmap=colormap, norm=norm, linewidths=0)


    f3.tight_layout()

    fig.tight_layout()
    plt.show()        
