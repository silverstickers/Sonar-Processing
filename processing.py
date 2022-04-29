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


    with serial.Serial(port='COM4', baudrate=115200, timeout=4) as arduino:
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data

        while True:
            zeroCount = 0
            while True:
                value = arduino.read(1)[0]
                if value == 1:
                    zeroCount += 1
                else:
                    zeroCount = 0
                if zeroCount > 48:
                    print("Looking for 50 ones in a row, found: {} ones in a row".format(zeroCount))
                if zeroCount == 50:
                    break
            
            message = arduino.readline()
            print(message.decode("utf-8"))
            message = arduino.readline()
            print(message.decode("utf-8"))
            message = arduino.readline()
            print(message.decode("utf-8"))
            
            bytes_per_array = int.from_bytes(arduino.read(4), byteorder='little')
            print("Bytes per array: {}".format(bytes_per_array))
            print(arduino.readline().decode("utf-8"))
            start_time = timeit.default_timer()
            array1 = arduino.read(bytes_per_array)
            array2 = arduino.read(bytes_per_array)
            elapsed = timeit.default_timer() - start_time

            

            print("Timed the serial read of 20kB. Elapsed time: {:2.3f} ms".format(elapsed * 1000))
            # print(int.from_bytes(array1[0:2], byteorder='little'))
            # print(int.from_bytes(array1[2:4], byteorder='little'))
            # print(int.from_bytes(array2[0:2], byteorder='little'))
            # print(int.from_bytes(array2[2:4], byteorder='little'))
            
            
            
            start_time = time.perf_counter_ns()
            array1 = convertToInt(array1)
            array2 = convertToInt(array2)
            elapsed = time.perf_counter_ns() - start_time
            print("Converted {} bytes into {} integers in {:2.3f} ms".format(bytes_per_array*2, bytes_per_array, elapsed * 1e-6))

            print(array1[0])
            print(array1[1])
            print(array2[0])
            print(array2[1])

            

            xarr = range(len(array1))

            ax1.clear()
            ax1.plot(xarr, array1)
            ax1.plot(xarr, array2)

            plt.pause(0.001)

            message = arduino.readline()
            print(message.decode("utf-8"))

            print("\n\n")
