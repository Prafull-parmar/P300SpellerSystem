import time
import os
import string
import datetime
import threading
from threading import Thread
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
from random import shuffle
from pylsl import StreamInfo, StreamOutlet, resolve_stream, StreamInlet
import preprocessing  as pre
import classifier as cf
import socket
import json
import pandas as pd
import pickle
import numpy as np
import statistics
import copy

from math import ceil
from math import floor as fl
from eeg_stream import receive_eeg_inlet
from eeg_stream import send_marker_outlet

# TCP/IP communication
skt = None
# port to connect to server
port = 12345

###############Timing vars##################
Epoch = 6
curr_Epoch = 0
set_time = 150
unset_time = 100
sample_count = int(Epoch * (set_time + unset_time) / 2) * 12

############### GLOBAL VARIABLES ###################
start_simulation = False
trainMode = False
testMode = False
isSubjectMode = False
start_time_f = 0
start_time_ts = 0
stop_time_f = 0
stop_time_ts = 0
index = -1
preprocess = False

train_text = "P300"
subject_name = ""
filename = ""
classify = False
current_index = 0
inlet = None
outlet = None
eegFile = None
eeg_thread = None
threadList = []

ascii_value = 0
marker_list = {}
marker_dict = {}
timestamp = None
eeg_data = None

############ Classifier #############
sampler_type = 0
dataset_type = 2
clf_type = 1

clf = []

path = os.path.join(os.getcwd(), 'Data')

baseline = None
baseline_data = None
# Fixed seed data
subject_train = ['O', 'I', '2', 'H', 'M', 'C', 'B', 'Q', 'S', '8', 'A', 'G', '7', 'Z', '4', 'L',
                 'Y', 'W', 'T', 'R', '6', 'F', 'K', '3', '0', 'X', 'J', 'U', 'D', 'V', '1', '5',
                 'E', '9', 'P', 'N']

# List to hold the random row and columns
List = []
for i in range(1, 13):
    List.append(i)
shuffle(List)
print(List)


############################################MENU Command Related Functions#####################################

####################### Training ################################
def do_training(isSubject):
    global trainMode, testMode

    if (start_simulation == True):
        messagebox.showwarning("Alert", "You Cannot switch modes currently!")
        return
    global trainMode, isSubjectMode
    trainMode = True
    testMode = False
    isSubjectMode = isSubject
    if (isSubjectMode is True):
        pred_label1.configure(text="Subject Name")
        pred_Text.configure(state="normal")
        pred_Text.delete('1.0', END)

    else:
        pred_label1.configure(text="Training Text")
        pred_Text.configure(state="normal")
    pred_label2.configure(text="Training Letter")
    pred_label3.configure(text="")


####################### Testing ################################
def do_testing():
    global skt, sample_count, unset_time, set_time

    if (start_simulation == True):
        messagebox.showwarning("Alert", "You Cannot switch modes currently!")
        return
    global trainMode, testMode
    trainMode = False
    testMode = True
    if skt is None:
        skt = socket.socket()
        skt.connect(('127.0.0.1', port))
        skt.send(str(sample_count).encode())
        temp = skt.recv(1024).decode()
        temp = int((set_time + unset_time) / 2)
        print(temp)
        skt.send(str(temp).encode())
        temp = skt.recv(1024).decode()
        skt.send(str(Epoch).encode())

    pred_label1.configure(text="Predicted Text")
    if (pred_Text["state"] == "disabled"):
        pred_Text.configure(state="normal")
    pred_Text.delete('1.0', END)
    pred_Text.configure(state="disabled")
    pred_label2.configure(text="Predicted Letter")
    pred_label3.configure(text="")


def choose_sampler(type):
    global sampler_type
    sampler_type = type
    print(f"Sampler_type: {type}")


def choose_data(type):
    global dataset_type
    dataset_type = type
    print(f"Dataset_type: {type}")
    create_directory(dataset_type)


def check_classifier(type):
    global clf, clf_type
    clf_type = type

    if (len(clf) == 0):
        clf = cf.wrapper(dataset_type, sampler_type, '', clf_type)
    else:
        select_message_box(4)


def settings_update(case):
    global sample_count
    if (start_simulation == True):
        return
    if case == 1:
        global Epoch
        res = simpledialog.askinteger("INPUT", "Enter number of Epochs.", minvalue=1, maxvalue=15)
        if res is not None:
            Epoch = res
        print("Update: Epoch= {}".format(Epoch))
    elif case == 2:
        global set_time
        res = simpledialog.askinteger("INPUT", "Enter set time in ms", minvalue=50, maxvalue=500)
        if res is not None:
            set_time = res
        print("Update: set_time= {}".format(set_time))
    elif case == 3:
        global unset_time
        res = simpledialog.askinteger("INPUT", "Enter unset time in ms", minvalue=50, maxvalue=500)
        if res is not None:
            unset_time = res
        print("Update: unset_time= {}".format(unset_time))

    sample_count = int(Epoch * (set_time + unset_time) / 2) * 12


############################################################################################################
def create_directory(cmd):
    global clf

    if (cmd == 1):
        loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Combined'))
        if not os.path.exists(loc):
            os.makedirs(loc)
    elif (cmd == 2):
        loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Session'))
        if not os.path.exists(loc):
            os.makedirs(loc)
    elif (cmd == 3):
        loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Subject'))
        if not os.path.exists(loc):
            os.makedirs(loc)


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


def baseline_creation(command):
    global outlet, unset_time, set_time, baseline, baseline_data, inlet
    sample_count = int(((set_time + unset_time) / 2) * 12)
    if (command == 'start'):
        create_lsl_stream()
        response = messagebox.askyesno("Success", f"Start Baseline recording?", parent=root)
        if (response):
            # send.....Marker
            baseline_data = np.empty(shape=[0, 8])
            counter = 0
            while counter < sample_count:
                arr = np.ones((1, 8))
                sample, ts = inlet.pull_sample()
                arr = arr * sample
                sample = arr
                baseline_data = np.concatenate((baseline_data, sample), axis=0)
                # np.savetxt(eeg_data,arr,delimiter=',')
                counter += 1
            inlet = None
            print(baseline_data)
            baseline = np.mean(baseline_data, axis=0)
            print("Baseline mean : ", baseline_data)
            messagebox.showinfo("Success", f"Baseline recording collected.", parent=root)


################### MessageBox ########################

def select_message_box(type):
    global preprocess, filename, classify, clf, dataset_type, sampler_type, path
    global clf_type
    if (type == 1):
        preprocess = messagebox.askokcancel("Success", "Training has been Completed!\nWould you like to preprocess ?",
                                            parent=root)
    elif (type == 2):
        classify = messagebox.askyesno("Success", f"File {filename} created!\nWould you like to train a classifier?",
                                       parent=root)

        if classify is True:
            print("Doing classification..........")
            if (dataset_type == 4):
                curr_path = os.path.join(path, "Normal_training\\" + filename + ".csv")
                clf = cf.wrapper(dataset_type, sampler_type, curr_path, clf_type)
            else:
                clf = cf.wrapper(dataset_type, sampler_type, "", clf_type)
    elif (type == 3):
        preprocess = messagebox.askokcancel("Success",
                                            f"Training for {train_text} been Completed!\nWould you like to preprocess ?",
                                            parent=root)

    elif (type == 4):
        response = messagebox.askyesno("Success",
                                       f"Classifier is already trained.\nWould you like to train classifier again?",
                                       parent=root)
        if (response):
            clf = cf.wrapper(dataset_type, sampler_type, "", clf_type)
        else:
            messagebox.showinfo("Alert", "Using the existing Classifier!!")


######################################

################# Sending Marker To server #######################
def create_marker_dict():
    global marker_dict, marker_list

    if testMode is True:
        marker_dict = {str(0): marker_list}

        serialized_dict = copy.deepcopy(marker_dict)
        print("Marker Dict")
        print(marker_dict)
        marker_list.clear()
        marker_dict.clear()
        print("Serialized Dict")
        print(serialized_dict)
        return serialized_dict


############## EEG Stream Controller ##############################


def create_file(subject_name):
    global filename, isSubjectMode, train_text
    global set_time, unset_time, Epoch, baseline
    print("In preprocessing ....")
    dest_path = os.path.join(os.getcwd(), 'Data')

    if isSubjectMode is True:

        filename = pre.create_filename(subject_name, "csv")
        print(filename)
        dest_path = os.path.join(dest_path, subject_name)
        pre.csv_preprocessing(filename, subject_name, int(set_time + unset_time) / 2)
        print("Doing Re-refrencing of electrodes:")
        pre.re_refrencing_electrodes('average', dest_path, str(filename))
        print("Doing Baseline_correction")
        pre.baseline_correction(baseline, dest_path, str(filename))

        print("Doing Filtering:")
        pre.eeg_filter(0.5, 15, dest_path, str(filename))

        print("Doing EEG ERP Averaging:")
        pre.erp_trial_averaging(dest_path, str(filename), Epoch, False)

    else:
        local_folder = "Word_Training"
        filename = pre.create_filename(local_folder, "csv", False, train_text)
        print(filename)
        dest_path = os.path.join(dest_path, local_folder)
        pre.csv_preprocessing(filename, local_folder, int(set_time + unset_time) / 2)

        print("Doing Re-refrencing of electrodes:")
        pre.re_refrencing_electrodes('average', dest_path, str(filename))

        print("Doing Baseline_correction")
        pre.baseline_correction(baseline, dest_path, str(filename))

        print("Doing Filtering:")
        pre.eeg_filter(0.5, 15, dest_path, str(filename))

        print("Doing EEG ERP Averaging:")
        pre.erp_trial_averaging(dest_path, str(filename), Epoch, False)
    select_message_box(2)


'''
Note:
index: 
 it is the index of the box under consideration

type:
    values =["CC","RBox"]
    CC->ColourChange 
    RBox-> Result Box
'''


def set(index, type):
    ## Sending the marker
    global set_time, ascii_value, box
    font_size = 28
    font_type = 'Times New Roman'
    # vec = []
    if (type == 'CC'):

        if (index < 7):
            index1 = (index - 1) * 6 + 1

            for i in range(index1, index1 + 6):
                box[i].config(fg="white", font=(font_type, font_size, 'bold'))

            print("row: ", index)

        else:
            index1 = index - 6
            for i in range(index1, index1 + 31, 6):
                box[i].config(fg="white", font=(font_type, font_size, 'bold'))
            print("col: ", index1)

    elif (type == "RBox"):
        ascii_value = ord(str(box[index].cget('text'))[0])
        box[index].config(background="blue", font=(font_type, font_size, 'bold'))
        print("index received : {}".format(index))


def unset(index, type):
    global unset_time, trainMode, box, List
    font_size = 20
    font_type = 'Times New Roman'
    if (type == "CC"):
        if (index < 7):
            index1 = (index - 1) * 6 + 1
            for i in range(index1, index1 + 6):
                box[i].config(fg="grey", font=(font_type, font_size))
            print("row ", index, " returned")
        else:
            index1 = index - 6
            for i in range(index1, index1 + 31, 6):
                box[i].config(fg="grey", font=(font_type, font_size))
            print("col ", index1, " returned")

        # intensification delay!
        root.after(unset_time, send_marker, index)
    elif (type == "RBox"):
        box[index].config(background="black", font=(font_type, font_size))
        if trainMode:
            row = int((index - 1) / 6) + 1
            col = (index - 1) % 6 + 7
            shuffle(List)
            while abs(List.index(row) - List.index(col)) not in range(4, 8):
                shuffle(List)
            change_color(0)
        else:
            shuffle(List)

    # intensification delay! Its useless now as function does not wait for it!(You can try to
    # add it anyways and see if its effective)


def result_display(row, col):
    global pred_Text, k_index, k_list
    global marker_dict, marker_list
    ind = (row - 1) * 6 + col - 6

    letter = str(box[ind].cget('text'))
    print(f'row :{row}, colum: {col} letter: {letter}')

    Text = pred_Text.get(1.0, "end-1c")
    Text += letter
    pred_Text.configure(state="normal")
    pred_Text.delete('1.0', END)
    ######kaam baki hai
    pred_Text.insert(END, Text)
    pred_Text.configure(state="disabled")
    pred_label3.configure(text=letter)
    print("Row: {}, Column: {} value: {} ".format(row, col, letter))
    set(ind, "RBox")
    state_controller()
    marker_list.clear()
    marker_dict.clear()
    root.after(10000, unset, ind, "RBox")


def eeg_stream_controller(cmd):
    global eeg_data, clf, sampler_type, dataset_type, start_simulation
    if (start_simulation == False):
        return
    if cmd is 1:

        print("Sending Start....")
        skt.send("Start".encode())
        while True:

            if skt.recv(1024).decode() == "Ok":
                print("Connection established with the eeg stream reciever.")
                break
            else:
                continue
            # print("Couldnt connect to eeg stream receiver !!!!!!!!!!")

        change_color(0)
    elif cmd is 2:
        skt.send("Stop".encode())
        while True:
            if skt.recv(1024).decode() != "Stopped":
                skt.send("Stop".encode())
                continue
            else:
                print("Received Stopped ")
                break
        serialized_dict = create_marker_dict()

        ################## For saving serialized_dict to a file ################
        filename = os.path.join('./Pickle', 'marker_dict')

        outfile = open(filename, 'wb')
        pickle.dump(serialized_dict, outfile)
        outfile.close()
        ########################################################################

        if skt:
            # skt.send(serialized_dict)
            skt.send("Start Reading".encode())
            while True:
                cmd = skt.recv(1024).decode()
                if cmd == "Markers Received":
                    print("Markers Sent")
                    break
                else:
                    continue

        else:
            print("Couldn't recieve ack of sent  marker dictionary")

        while True:
            if skt.recv(1024).decode() == "Dataframe sent":
                data_arr = pd.read_csv(os.path.join(os.getcwd(), 'temp.csv'))
                eeg_data = data_arr.iloc[:, :].values
                skt.send("Dataframe Received".encode())
                print("Dataframe recieved")
                break
            else:
                continue

        print("Received EEG data is :\n")
        print(eeg_data)

        row = []
        column = []
        i = 0
        for classifier in clf:
            r, c = cf.predictor(eeg_data, classifier, '', True)
            i += 1
            ind = (r - 1) * 6 + c - 6
            if ind not in range(1, 37):
                ind = 36
            letter = str(box[ind].cget('text'))
            print("Clf [{}]  Row: {} Column : {} Letter : {} ".format(i, r, c, letter))
            row.append(r)
            column.append(c)

        row_mode = statistics.mode(row)
        column_mode = statistics.mode(column)
        result_display(row_mode, column_mode)


#######################################################################33


############### For sending Marker to NIC and To Server ###########################
def send_marker(index):
    global ascii_value, curr_Epoch

    timestamp = int(time.time() * 100)
    if testMode:
        marker_list[timestamp] = index
    else:
        vec = []
        if (ascii_value < 65):
            temp = ascii_value - 22
        else:
            temp = ascii_value - 65

        col = temp % 6 + 7
        row = fl(temp / 6) + 1
        curr_marker = index * 1000000 + (row + 10) * 10000 + (col + 10) * 100 + (curr_Epoch + 10)
        vec.append(curr_marker)
        outlet.push_sample(vec)
        print("Now sending marker: \t" + str(curr_marker) + "\n")

    # create_marker_dict()
    root.after(0, change_color, curr_Epoch)


####################################################################################


# current_epoch -> local parameter to check total epochs completed currently
def change_color(current_epoch):
    global start_simulation, index, Epoch, train_text, current_index
    global trainMode, curr_Epoch
    global set_time

    # marker_dict= {}

    if (start_simulation == False):
        return
    # if(index>=0):
    #     unset(List[index],"CC")
    index = (index + 1)
    if (index >= 12):
        current_epoch += 1
        index = 0
        # shuffle(List)
        if (current_epoch >= Epoch):
            if (trainMode is True):
                index = -1
                letter_trainer()
                return
            else:
                index = -1
                eeg_stream_controller(2)

                current_epoch = 0
                return

                # eeg_stream_controller("Start")
                # marker_dict = {0: marker_list}
                # serialized_dict = json.dumps(marker_dict)
                # skt.sendall(serialized_dict)

    set(List[index], "CC")
    # intensification time
    curr_Epoch = current_epoch
    if (index >= 0):
        root.after(set_time, unset, List[index], "CC")


def letter_trainer():
    global current_index, train_text, subject_train, isSubjectMode, trainMode, preprocess, subject_name
    global start_simulation, filename
    if (start_simulation == False or trainMode == False):
        return
    if (isSubjectMode is True):
        subject_name = train_text

        if (current_index + 1 < len(subject_train)):
            current_index += 1
            letter = subject_train[current_index]
            pred_label3.configure(text=letter)
            if (letter.isalpha()):
                box_num = ord(letter) - ord('A') + 1
            elif (letter.isnumeric()):
                box_num = ord(letter) - ord('0') + 1 + 26
            else:
                box_num = 36
            # resultDisplay(0,ord(temp)-ord('A')+1)
            set(box_num, "RBox")
            root.after(5000, unset, box_num, "RBox")
        else:
            current_index = -1
            state_controller()
            select_message_box(1)
            if (preprocess is True):
                create_file(subject_name)
                preprocess = False
    else:
        if (current_index + 1 < len(train_text)):
            current_index += 1
            letter = train_text[current_index].upper()
            pred_label3.configure(text=letter)
            if (letter.isalpha()):
                box_num = ord(letter) - ord('A') + 1
            elif (letter.isnumeric()):
                box_num = ord(letter) - ord('0') + 1 + 26
            else:
                box_num = 36
            set(box_num, "RBox")
            root.after(5000, unset, box_num, "RBox")
        else:
            current_index = -1
            state_controller()
            select_message_box(3)
            if (preprocess is True):
                create_file(subject_name)
                preprocess = False


def update_time():
    global start_time_f, stop_time_f

    starttime.configure(text=start_time_f)
    stoptime.configure(text=stop_time_f)


def state_controller():
    global start_simulation, index, current_index, start_time_ts, start_time_f
    global stop_time_ts, stop_time_f, eeg_thread, testMode, trainMode, clf

    if testMode == True and len(clf) == 0:
        start_simulation = False
        messagebox.showinfo("Alert", "Classifier is not trained !!!")
        return
    if (start_simulation == False):
        start_time_ts = time.time()
        start_time_f = time.strftime("%H:%M:%S %p")
        start_simulation = True
        button.configure(text='Stop')
    else:
        start_simulation = False
        stop_time_ts = time.time()
        stop_time_f = time.strftime("%H:%M:%S %p")
        update_time()
        if trainMode:
            unset(List[index], "CC")
        index = -1
        current_index = -1
        button.configure(text='Start')


def start_speller():
    global start_simulation, index, start_time_ts, stop_time_ts, subject_name
    global start_time_f, stop_time_f
    global trainMode, train_text, current_index, port, skt

    if trainMode is False and testMode is False:
        messagebox.showinfo("Alert", "Please select a mode!")
        return
    train_text = pred_Text.get(1.0, "end-1c")

    if (trainMode is True and isSubjectMode is True and train_text is ""):
        messagebox.showinfo("Alert", "Please enter your name!")
    elif (trainMode is True and train_text is ""):
        messagebox.showinfo("Alert", "Please enter text for training!")
    else:
        state_controller()

        if (trainMode is True):
            # auto-recovery from last letter before stopping.... do not update current_index then!
            current_index = -1
            # pred_label3.config(text=train_text[0].upper())
            letter_trainer()
        else:
            eeg_stream_controller(1)


################################ Main Program starts here.... ####################################   

root = Tk()
menubar = Menu(root)
root.config(menu=menubar, bg="#666666")

############# FILE Menu #####################
filemenu = Menu(menubar, tearoff=0)
submenu = Menu(filemenu, tearoff=0)
menubar.add_cascade(label='Mode', menu=filemenu)
submenu.add_command(label='Subject Training Mode', command=lambda: do_training(True))
submenu.add_command(label='Word Training Mode', command=lambda: do_training(False))
filemenu.add_cascade(label="Training Mode", menu=submenu, underline=0)
filemenu.add_command(label='Testing Mode', command=lambda: do_testing())
filemenu.add_separator()
filemenu.add_command(label='Quit', command=sys.exit)

############# MODEL Menu ###################

model = Menu(menubar, tearoff=0)
toggleVar = IntVar()

sample_menu = Menu(model, tearoff=0)
sample_menu.add_radiobutton(label='None', variable=toggleVar, value=1, command=lambda: choose_sampler(0))
sample_menu.add_radiobutton(label='SMOTEENN', variable=toggleVar, value=1, command=lambda: choose_sampler(1))
sample_menu.add_radiobutton(label='SMOTETomek', variable=toggleVar, value=2, command=lambda: choose_sampler(2))
sample_menu.add_radiobutton(label='RandomUnderSampler', variable=toggleVar, value=3, command=lambda: choose_sampler(3))
sample_menu.add_radiobutton(label='RandomOverSampler', variable=toggleVar, value=4, command=lambda: choose_sampler(4))
sample_menu.add_radiobutton(label='ADASYN', variable=toggleVar, value=5, command=lambda: choose_sampler(5))
model.add_cascade(label='Sampler', menu=sample_menu)

data_menu = Menu(model, tearoff=0)
tVar = IntVar()

data_menu.add_radiobutton(label='Combined', variable=tVar, value=1, command=lambda: choose_data(1))
data_menu.add_radiobutton(label='Session-wise', variable=tVar, value=2, command=lambda: choose_data(2))
data_menu.add_radiobutton(label='Subject-wise', variable=tVar, value=3, command=lambda: choose_data(3))
model.add_cascade(label='Dataset Type', menu=data_menu)

menubar.add_cascade(label='Model', menu=model)
###############################################

######################Classifier Menu#######################

classifier = Menu(menubar, tearoff=0)
toggle = IntVar()
classifier.add_radiobutton(label='Saved Model', variable=toggle, value=3, command=lambda: check_classifier(0))
classifier.add_radiobutton(label='Neural Network', variable=toggle, value=1, command=lambda: check_classifier(1))
classifier.add_radiobutton(label='SVM', variable=toggle, value=2, command=lambda: check_classifier(2))

menubar.add_cascade(label='Classifier', menu=classifier)

settings = Menu(menubar, tearoff=0)
settings.add_command(label="Epoch", command=lambda: settings_update(1))
settings.add_command(label="Set_time", command=lambda: settings_update(2))
settings.add_command(label="Unset_time", command=lambda: settings_update(3))
settings.add_command(label="Baseline", command=lambda: baseline_creation('start'))
menubar.add_cascade(label='Settings', menu=settings)

####################### Creating the CANVAS Structure #################################
box = dict()
import math

width = math.ceil(root.winfo_screenwidth() / 6)
height = math.ceil(root.winfo_screenheight() / 7.75)
roots = Canvas(root, height=6 * height, width=6 * width, bg="#666666", highlightthickness=0, bd=0)
frame = {}

##======================Creating the 6x6 Grid of P300 Speller=====================##
for r in range(1, 7):
    for c in range(1, 7):
        if (6 * (r - 1) + c <= 26):
            frame[6 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
            frame[6 * (r - 1) + c].pack_propagate(0)  # Stops child widgets of label_frame from resizing it
            box[6 * (r - 1) + c] = Label(frame[6 * (r - 1) + c], text=chr(64 + 6 * (r - 1) + c), borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
            box[6 * (r - 1) + c].pack(fill="both", expand=True, side='left')
            frame[6 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)

        else:
            frame[6 * (r - 1) + c] = Frame(roots, width=width, height=height, bg="white")
            frame[6 * (r - 1) + c].pack_propagate(0)
            box[6 * (r - 1) + c] = Label(frame[6 * (r - 1) + c], text=6 * (r - 1) + c - 27, borderwidth=0,
                                         background="black", width=width, height=height, fg="grey",
                                         font=("Courier", 19))
            box[6 * (r - 1) + c].pack(fill="both", expand=True, side='left')
            frame[6 * (r - 1) + c].place(x=(c - 1) * width, y=(r - 1) * height)

roots.pack(fill="both", expand=True)
##===============================================================================##

##========================Creating the Control Panel below the grid=======================##
root7 = Canvas(root, bg="#666666", highlightthickness=0, bd=0)

pred_label1 = Label(root7, text="Predicted Text", fg="Black", font=("Times New Roman", 18), bg="#666666", width="12")
pred_label1.grid(row=0, column=0)

pred_Text = Text(root7, height=1, width=15, font=("Times New Roman", 18), state="disabled")
pred_Text.grid(row=1, column=0)

pred_label2 = Label(root7, text="Predicted Letter", fg="Black", font=("Times New Roman", 18), bg="#666666", width="12",
                    padx="10")
pred_label2.grid(row=0, column=1)

pred_label3 = Label(root7, text="A", fg="Black", font=("Times New Roman", 18), bg="#666666", padx="10", pady="10")
pred_label3.grid(row=1, column=1)

button = Button(root7, text='Start', width=10, height=1, fg="#666666", bg="#222222", font=("Times New Roman", 20),
                command=start_speller, activebackground="#666666", activeforeground="black")
button.grid(rowspan=2, row=0, column=2)

start_label = Label(root7, text='Start Time:', fg='black', font=("Times New Roman", 17), bg="#666666", padx="40")
start_label.grid(row=0, column=3)

starttime = Label(root7, text='00:00:00 00', fg='black', font=("Times New Roman", 17), bg="#666666", width="10")
starttime.grid(row=0, column=4)

stop_label = Label(root7, text='Stop Time: ', fg='black', font=("Times New Roman", 17), bg="#666666", padx="40")
stop_label.grid(row=1, column=3)

stoptime = Label(root7, text='00:00:00 00', fg='black', font=("Times New Roman", 17), bg="#666666", width="10")
stoptime.grid(row=1, column=4)

root7.pack()

outlet = send_marker_outlet()
root.title("P300 Speller")
root.mainloop()
