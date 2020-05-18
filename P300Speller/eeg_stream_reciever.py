########  Read from Enobio EEG Data  ########

from pylsl import StreamInlet, resolve_stream
import os
from datetime import datetime
import csv
import socket
import json
import numpy as np
import preprocessing as pre
import time
import pandas as pd
import pickle
from pprint import pprint
from threading import Thread

####### GLOBAL variables ###########
cskt = None
marker_dict = {}
eeg_data = None
inlet = None
ReadingData = False
sample_count = 0
records_per_marker = 0
Epochs = 0


class Client_thread(Thread):

    def __init__(self, ip, port):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        print("[+] New server socket thread started for " + ip + ":" + str(port))

    def run(self):
        global sample_count, cskt, ReadingData
        sample_count = int(cskt.recv(1024).decode())
        print("Starting reading EEG data.......")
        while True:
            if ReadingData:
                continue
            else:
                read_eeg()


def read_eeg():
    global marker_dict, eeg_data, inlet, ReadingData, sample_count
    global records_per_marker
    ReadingData = True
    print("In  read_EEG.....")
    print("Listening for Start Command....")
    ##   Starting the connection.
    while True:
        if cskt.recv(1024).decode() == "Start":
            print("Sending Ok....")
            cskt.send("Ok".encode())
            break
        else:
            continue
    print("Creating Dataframe...")
    eeg_data = np.empty(shape=[0, 10])
    counter = 0
    while counter <= sample_count:

        arr = np.ones((1, 10))
        sample, ts = inlet.pull_sample()
        timestamp = int(time.time() * 100)
        # Appending Marker with 0 default value
        sample.append(0)
        sample.append(timestamp)
        arr = arr * sample
        sample = arr
        eeg_data = np.concatenate((eeg_data, sample), axis=0)
        counter += 1
        if counter % 300 == 0:
            print("highlighted")

    pprint("************  Done collecting data  ************")
    while True:
        if cskt.recv(1024).decode() != "Stop":

            continue
        else:
            print("Sending Stopped ....")
            cskt.send("Stopped".encode())
            break
    print(eeg_data)

    marker_dict.clear()
    filename = os.path.join('./Pickle', 'marker_dict')
    while True:
        if cskt.recv(1024).decode() == "Start Reading":
            print("Reading marker dictionary file........")
            infile = open(filename, 'rb')
            marker_dict = pickle.load(infile)
            pprint(marker_dict)
            infile.close()
            marker_dict = marker_dict['0']
            cskt.send("Markers Received".encode())
            pprint(marker_dict)
            break
        else:
            continue

    present_marker = marker_dict[list(marker_dict.keys())[0]]
    marker_timestamps = list(map(int, list(marker_dict.keys())))

    while (int(eeg_data[0, 9]) in range(0, marker_timestamps[0])):
        eeg_data = np.delete(eeg_data, (0), axis=0)
    print("Deletion done.....")
    print(eeg_data)
    current_marker = marker_dict[marker_timestamps[0]]
    current_timestamp = marker_timestamps[1]
    counter = 0
    i = 0
    while i < eeg_data.shape[0]:
        if (int(eeg_data[i, 9]) < current_timestamp):
            if (counter < records_per_marker):
                eeg_data[i, 8] = current_marker
                counter += 1
            else:
                eeg_data = np.delete(eeg_data, (i), axis=0)

        elif (int(eeg_data[i, 9]) >= current_timestamp):
            # Making equal no. of records for all markers
            if counter < records_per_marker and i > 0:
                sample = np.ones((1, 10)) * eeg_data[i - 1, :]
                while counter < records_per_marker:
                    eeg_data = np.insert(eeg_data, i, sample, axis=0)
                    counter += 1
                    i += 1
            counter = 0
            if marker_timestamps.index(current_timestamp) + 1 < len(marker_timestamps):
                current_marker = marker_dict[current_timestamp]
                current_timestamp = marker_timestamps[marker_timestamps.index(current_timestamp) + 1]
                eeg_data[i, 8] = current_marker
            else:
                current_marker = marker_dict[current_timestamp]

                xlen = min(i + records_per_marker, eeg_data.shape[0])
                for j in range(i, xlen):
                    eeg_data[j, 8] = current_marker
                # current_timestamp = eeg_data[xlen ,9]
                if (xlen < eeg_data.shape[0]):
                    while (xlen != eeg_data.shape[0] - 1):
                        eeg_data = np.delete(eeg_data, (xlen), axis=0)
                break
        i += 1

    ###############################################
    print("***********Processing done!**********")
    np.set_printoptions(threshold=np.nan)
    print(eeg_data)
    dataframe = np.array(eeg_data[:, :9])
    print(dataframe)

    ####Preprocessing eeg_data
    fd = open("temp.csv", 'wb')
    np.savetxt(fd, dataframe, delimiter=",")
    fd.close()

    print("Filterting data:")
    pre.eeg_filter(0.5, 10, os.getcwd(), "temp")

    print("ERP Averageing:")
    pre.test_erp_trial_averaging(os.getcwd(), "temp", Epochs, records_per_marker)
    file = pd.read_csv(os.path.join(os.getcwd(), "temp.csv"), header=None)
    print(file)
    cskt.send("Dataframe sent".encode())
    while True:
        if cskt.recv(1024).decode() == "Dataframe Received":
            print("Dataframe Sent ")
            cskt.send("Done".encode())
            break
        else:
            cskt.send("Dataframe sent".encode())
            continue

    print("Completed.....")
    ReadingData = False
    inlet = None


def create_lsl_stream():
    # Creating an EEG Stream using pylsl
    global inlet
    print("Creating EEG stream ....")
    stream_name = 'NIC'
    streams = resolve_stream('type', 'EEG')
    try:
        for i in range(len(streams)):

            if (streams[i].name() == stream_name):
                index = i
                print("NIC stream available")

        print("Connecting to NIC stream... \n")
        inlet = StreamInlet(streams[index])
        print("Connected to EEG Inlet .....")
    except NameError:
        print("Error: NIC stream not available\n\n\n")


if __name__ == "__main__":

    # Creating a  Server Socket 
    s = socket.socket()
    print("Socket successfully created")
    port = 12345
    s.bind(('', port))
    print("socket binded to %s" % (port))
    # Listen for atmost 5 connections
    s.listen(5)
    print("socket is listening")
    # create a lsl stream
    while True:
        # Establish connection with client.
        cskt, addr = s.accept()
        print(f'Got connection from {addr}')
        sample_count = int(cskt.recv(1024).decode())
        print("sample count: {}".format(sample_count))
        cskt.send(str("Ok").encode())
        records_per_marker = int(cskt.recv(1024).decode())
        print("records_per_marker:{}".format(records_per_marker))
        cskt.send(str("Ok").encode())
        Epochs = int(cskt.recv(1024).decode())
        print("Epochs:{}".format(Epochs))
        print("Starting reading EEG data.......")
        while True:
            if ReadingData:
                continue
            else:
                create_lsl_stream()
                read_eeg()
