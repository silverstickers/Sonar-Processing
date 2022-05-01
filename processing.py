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
                print("Bytes per array: {}".format(bytes_per_array))
                start_time = timeit.default_timer()
                array1 = arduino.read(bytes_per_array)
                array2 = arduino.read(bytes_per_array)
                elapsed = timeit.default_timer() - start_time

                
                print("Timed the serial read. Elapsed time: {:2.3f} ms".format(elapsed * 1000))
                
                
                start_time = time.perf_counter_ns()
                array1 = convertToInt(array1)
                array2 = convertToInt(array2)
                elapsed = time.perf_counter_ns() - start_time
                print("Converted {} bytes into {} integers in {:2.3f} ms".format(bytes_per_array*2, bytes_per_array, elapsed * 1e-6))
                

                xarr = range(len(array1))

                if (max(array1) < 4100 and max(array2) < 4100):
                    ax1.clear()
                    ax1.plot(xarr, array1)
                    ax1.plot(xarr, array2)

                    plt.pause(0.001)

                if filesWritten < 10:
                    with open('waveform_{}'.format(filesWritten), 'w') as f:
                        for a, b in zip(array1, array2):
                            f.write("{} {}\n".format(a, b))
                    filesWritten += 1

                message = arduino.readline()
                print(message.decode("utf-8"))

                print("\n")
            
            except UnicodeDecodeError:
                pass
