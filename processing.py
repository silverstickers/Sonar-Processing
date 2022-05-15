from socket import timeout
from textwrap import wrap
import serial
import time
import timeit
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convertToInt(array):
    # return [int.from_bytes(array[2*i:2*i+1], byteorder='little') \
            # for i in range((int)(bytes_per_array/2))]
    new_array = []
    for a in zip(array[::2], array[1::2]):
        new_array.append(int.from_bytes(a, byteorder='little'))
    return new_array


if __name__ == '__main__':
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1,1,1)
    plt.ion()
    plt.show(block=False)

    filesWritten = 0
    saveWaveforms = True


    with serial.Serial(port='COM4', baudrate=115200, timeout=5.0) as arduino:
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data

        while True:
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
                    print("Found {} {} in a row".format(count, pattern[patternPosition]))
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
                print("Converted {} bytes into {} integers in {:2.3f} ms".format(bytes_per_array*2, bytes_per_array, elapsed * 1e-6))
                

                xarr = range(len(arrays[0]))

                if (max(arrays[0]) < 4100 and max(arrays[1]) < 4100):
                    ax1.clear()
                    for arr in arrays:
                        ax1.plot(xarr, arr)

                    plt.pause(0.001)

                if filesWritten < 10:
                    with open('waveforms/30deg_waveform_{}'.format(filesWritten), 'w') as f:
                        for i in range(len(arrays[0])):
                            for j in range(len(arrays)):
                                f.write("{} ".format(arrays[j][i]))
                            f.write("\n")
                    filesWritten += 1
                else:
                    pass
                    # break

                message = arduino.readline()
                print(message.decode("utf-8"))

                print("\n")
            
            except UnicodeDecodeError:
                pass
