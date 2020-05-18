import csv
import os
from math import floor as fl
import fnmatch
import pandas as pd
import numpy as np
import mne
import math

####### Global variables #######
default_name = "Session.easy"
default_filename = "Session.easy"
subjectName = "Subject_01"
filename = ""


### Note: This function returns filename without extension

def create_filename(subject_name, fileType, isSubjectMode=True, train_text=''):
    path = os.path.join('./Data', subject_name)

    if not os.path.exists(path):
        os.makedirs(path)
    no = 1

    if isSubjectMode is True:

        while (os.path.exists(path + '/' + subject_name + "_Session_" + str(no) + "." + fileType)):
            no += 1
        open(path + '/' + subject_name + "_Session_" + str(no) + "." + fileType, 'a').close()
        file_name = subject_name + "_Session_" + str(no)

    else:
        while (os.path.exists(path + '/' + subject_name + "_" + train_text + "_" + str(no) + "." + fileType)):
            no += 1
        open(path + '/' + subject_name + "_" + train_text + "_" + str(no) + "." + fileType, 'a').close()
        file_name = subject_name + "_" + train_text + "_" + str(no)

    return file_name


def csv_preprocessing(filename, subjectname, no_records):
    global default_name, default_filename
    print("Starting csv_preprocessing:")
    src_path = os.path.join(os.getcwd(), 'Easy')
    dest_path = os.path.join(os.getcwd(), 'Data')
    dest_path = os.path.join(dest_path, subjectname)

    for fname in os.listdir(os.path.join(os.getcwd(), 'Easy')):
        if fnmatch.fnmatch(fname, '*Session.easy'):
            default_filename = fname
            break
    def_filename = os.path.splitext(default_filename)[0]

    if (os.path.isfile(os.path.join(src_path, default_filename))):

        # Renaming other file formats like .info and .edf
        if (os.path.isfile(os.path.join(src_path, def_filename + ".info"))):
            os.rename(os.path.join(src_path, def_filename + ".info"), os.path.join(src_path, filename + ".info"))
        if (os.path.isfile(os.path.join(src_path, def_filename + ".edf"))):
            os.rename(os.path.join(src_path, def_filename + ".edf"), os.path.join(src_path, filename + ".edf"))

        # print(filename)
        os.rename(os.path.join(src_path, default_filename), os.path.join(src_path, filename + ".easy"))

        dest_csv_file = str(filename) + ".csv"
        src_easy_file = str(filename) + ".easy"
        source_file = os.path.join(src_path, src_easy_file)
        print(source_file)

        dest_file = os.path.join(dest_path, dest_csv_file)

        with open(source_file, "r") as infile:
            prev_val = marker = asciivalue = temp = epoch = 0
            coordinates = []
            prev_row = 0
            ctr = -1
            reader = csv.reader(infile, dialect="excel-tab")
            with open(dest_file, "w") as outfile:
                writer = csv.writer(outfile, delimiter=',')
                for row in reader:

                    if (int(row[8], 10) > 0):
                        if ctr > 0 and ctr < no_records:
                            while ctr < no_records:
                                ctr = ctr + 1
                                writer.writerow(prev_row)
                        ctr = 0
                        prev_val = row[8]
                        temp = fl(int(prev_val) % 1000000)
                        marker = fl(int(prev_val) / 1000000)
                        r = fl(temp / 10000) - 10
                        temp %= 10000
                        c = fl(temp / 100) - 10
                        epoch = temp % 100 - 10
                        asciivalue = c * 100 + r
                        coordinates.clear()
                        coordinates.append(c)
                        coordinates.append(r)
                    else:
                        if (prev_val is 0):
                            continue

                    if (ctr < no_records):
                        row[8] = marker
                        row.append(asciivalue)
                        row.append(epoch)
                        ctr = ctr + 1
                        if (row[8] in coordinates):
                            row.append(1)
                        else:
                            row.append(0)
                        prev_row = row.copy()
                        writer.writerow(row)


def eeg_filter(min_freq, max_freq, path, filename):
    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)
    data = np.array(file.iloc[:, :8])
    data = data.transpose()
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['PO7', 'P3', 'Fz', 'Cz', 'Pz', 'P4', 'PO8', 'Oz']
    sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")

    # Filtering EEG Data....
    raw.filter(min_freq, max_freq, fir_design='firwin')
    raw.plot(n_channels=8, scalings='auto', show=True, block=True)

    data, time = raw[:]

    data = data.transpose()

    file.iloc[:, :8] = data

    fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, file, delimiter=",")
    fd.close()
    print("Filtering done!")


def re_refrencing_electrodes(re_ref, path, filename):
    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)

    data = np.array(file.iloc[:, :8])
    data = data.transpose()
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    ch_names = ['PO7', 'P3', 'Fz', 'Cz', 'Pz', 'P4', 'PO8', 'Oz']
    sfreq = 500
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1020")

    # Re-refrencing electrodes.......
    raw.set_eeg_reference(re_ref)

    data, time = raw[:]

    data = data.transpose()

    file.iloc[:, :8] = data

    fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, file, delimiter=",")
    fd.close()
    print("Re-refrencing done!")


def baseline_correction(baseline, path, filename):
    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)

    data = np.array(file.iloc[:, :8])

    for record in data:
        record -= baseline

    file.iloc[:, :8] = data

    fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, file, delimiter=",")
    fd.close()
    print("Baseline_correction done!")


def test_erp_trial_averaging(path, filename, Epochs, records):
    print("Starting ERP trail averaging:")

    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)

    batch = np.empty(shape=[0, 9])
    average = np.empty(shape=[0, 9])
    records = records * 12

    average = np.array(file.iloc[:records, :])
    current_epoch = 1
    for i in range(records, file.shape[0]):
        if (current_epoch < Epochs):
            temp = min(i + records, file.shape[0])
            batch = np.array(file.iloc[i:temp, :])

            while (average.shape[0] != batch.shape[0]):
                if (average.shape[0] < batch.shape[0]):
                    j = average.shape[0]
                    sample = np.ones((1, 9)) * (batch[j, :] * current_epoch)
                    average = np.concatenate((average, sample), axis=0)

                else:
                    j = batch.shape[0]
                    sample = np.ones((1, 9)) * (average[j, :] / current_epoch)
                    batch = np.concatenate((batch, sample), axis=0)
            i = temp
            current_epoch += 1
            # batch = batch/Epochs
            average = np.add(average, batch)
            batch = np.empty(shape=[0, 9])
    average = average / Epochs

    for i in range(average.shape[0]):
        if (average[i, 8] - math.floor(average[i, 8])) >= 0.5:
            average[i, 8] = math.ceil(average[i, 8])
        else:
            average[i, 8] = math.floor(average[i, 8])

    fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, average, delimiter=",")
    fd.close()
    print("Done with eeg_averaging")


def erp_trial_averaging(path, filename, epochs, create_file=False):
    print("Starting ERP trail averaging:")
    if (create_file):
        newfilename = os.path.splitext(filename)[0] + "_avg.csv"
        newfilepath = os.path.join(path, newfilename)
        open(newfilepath, 'a').close()
    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)

    # batch_size = 11
    eeg_data = np.empty(shape=[0, 13])
    batch = np.empty(shape=[0, 13])
    average = np.empty(shape=[0, 13])
    current_ascii = file.iloc[0, 10]
    current_epoch = file.iloc[0, 11]
    for i in range(file.shape[0]):
        if (file.iloc[i, 10] == current_ascii):
            arr = np.ones((1, 13))
            sample = file.iloc[i, :].values
            sample = arr * sample
            # sample[0,8]= 0
            if (file.iloc[i, 11] != current_epoch):

                if (average.shape[0] == 0):
                    average = batch
                else:
                    # Making both the numpy arrays of equal shape
                    while average.shape[0] != batch.shape[0]:
                        if (average.shape[0] < batch.shape[0]):
                            j = average.shape[0]
                            temp = np.ones((1, 13)) * (batch[j, :] * current_epoch)
                            average = np.concatenate((average, temp), axis=0)

                        else:
                            j = batch.shape[0]
                            temp = np.ones((1, 13)) * (average[j, :] / current_epoch)
                            batch = np.concatenate((batch, temp), axis=0)

                    # batch = batch
                    average = (np.add(average, batch))
                current_epoch = file.iloc[i, 11]
                batch = np.empty(shape=[0, 13])
            batch = np.concatenate((batch, sample), axis=0)


        elif (file.iloc[i, 10] != current_ascii):

            if (batch.shape[0] > (average.shape[0] / 2)):
                while average.shape[0] != batch.shape[0]:
                    if (average.shape[0] < batch.shape[0]):
                        j = average.shape[0]
                        temp = np.ones((1, 13)) * (batch[j, :] * current_epoch)
                        average = np.concatenate((average, temp), axis=0)

                    else:
                        j = batch.shape[0]
                        temp = np.ones((1, 13)) * (average[j, :] / current_epoch)
                        batch = np.concatenate((batch, temp), axis=0)
                average = (np.add(average, batch))
            average = average / epochs
            eeg_data = np.concatenate((eeg_data, average), axis=0)
            average = np.empty(shape=[0, 13])
            batch = np.empty(shape=[0, 13])
            current_ascii = file.iloc[i, 10]
            current_epoch = file.iloc[i, 11]
            arr = np.ones((1, 13))
            sample = file.iloc[i, :].values
            sample = arr * sample
            batch = np.concatenate((batch, sample), axis=0)

        # Ending condition....
        if (i == file.shape[0] - 1):
            while average.shape[0] != batch.shape[0]:
                if (average.shape[0] < batch.shape[0]):
                    i = average.shape[0]
                    temp = np.ones((1, 13)) * (batch[i, :] * current_epoch)
                    average = np.concatenate((average, temp), axis=0)

                else:
                    i = batch.shape[0]
                    temp = np.ones((1, 13)) * (average[i, :] / current_epoch)
                    batch = np.concatenate((batch, temp), axis=0)
            average = (np.add(average, batch)) / epochs

            eeg_data = np.concatenate((eeg_data, average), axis=0)

    # saving file
    if (create_file):
        fd = open(newfilepath, "wb")
    else:
        fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, eeg_data, delimiter=",")
    fd.close()
    print("Done with eeg_averaging")


def eeg_data_downsampling(path, filename, no_records, batch_size=10, create_file=False):
    if (create_file):
        newfilename = filename + "_stat.csv"
        newfilepath = os.path.join(path, newfilename)
        open(newfilepath, 'a').close()
    file = pd.read_csv(os.path.join(path, filename + ".csv"), header=None)

    # batch_size = 11
    rec_counter = 0
    counter = 0
    eeg_data = np.empty(shape=[0, 13])
    batch = np.empty(shape=[0, 13])
    current_marker = file.iloc[0, 8]
    for i in range(file.shape[0]):
        if (file.iloc[i, 8] == current_marker):
            arr = np.ones((1, 13))
            sample = file.iloc[i, :].values
            sample = arr * sample

            if (rec_counter < no_records):
                batch = np.concatenate((batch, sample), axis=0)
                counter += 1
                if (counter >= batch_size):
                    mean = np.mean(batch, axis=0).reshape(1, 13)
                    batch = np.empty(shape=[0, 13])
                    eeg_data = np.concatenate((eeg_data, mean), axis=0)
                    counter = 0
                    rec_counter += 1

        elif (file.iloc[i, 8] != current_marker):
            if (counter > 0 and rec_counter < no_records):
                mean = np.mean(batch, axis=0).reshape(1, 13)
                eeg_data = np.concatenate((eeg_data, mean), axis=0)
                batch = np.empty(shape=[0, 13])
            current_marker = file.iloc[i, 8]
            rec_counter = 0
            counter = 1
            arr = np.ones((1, 13))
            sample = file.iloc[i, :].values
            sample = arr * sample
            batch = np.concatenate((batch, sample), axis=0)

    # saving file
    if (create_file):
        fd = open(newfilepath, "wb")
    else:
        fd = open(os.path.join(path, filename + ".csv"), "wb")
    np.savetxt(fd, eeg_data, delimiter=",")
    fd.close()
    print("Done with eeg data downsampling")
