from pylsl import StreamInfo, StreamOutlet, resolve_stream, StreamInlet
import time

# Initializing all inlets and outlets used
inlet1 = None
inlet2 = None
outlet = None


################## lsl inlet for receiving EEG Data Stream ###########################
def receive_eeg_inlet():
    global inlet1, inlet2
    stream_name = 'NIC'
    print("Creating Inlet...")
    streams1 = resolve_stream('type', 'Markers')
    inlet1 = StreamInlet(streams1[0])
    streams2 = resolve_stream('type', 'EEG')
    inlet2 = StreamInlet(streams2[0])
    print("Created Inlet...")
    try:
        for i in range(len(streams1)):
            if (streams1[i].name() == stream_name):
                index = i
                print("NIC marker stream available")

            print("Connecting to NIC  marker stream... \n")
            inlet1 = StreamInlet(streams1[index])

        for i in range(len(streams2)):
            if (streams2[i].name() == stream_name):
                index = i
                print("NIC data stream available")

            print("Connecting to NIC data stream... \n")
            inlet2 = StreamInlet(streams2[index])

        return inlet1, inlet2

    except NameError:
        print("Error: NIC stream not available\n\n\n")


#####################################################################################

################### lsl outlet for sending marker stream ############################
def send_marker_outlet():
    ##Creating a marker stream
    global outlet

    print("Creating a new marker stream info...\n")
    info = StreamInfo('MyMarkerStream', 'Markers', 1, 0.01, 'int32', 'kalam151070018')

    print("Opening an outlet...\n")
    outlet = StreamOutlet(info)

    print("Sending data...\n")

    return outlet

#####################################################################################
