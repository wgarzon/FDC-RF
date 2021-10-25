import zmq
import time
import sys
import initialize
import socket
import main_FDC

port = "5557"
NoSites = 1
file = "mainc.ini"
opc = ""

#python3 mainControl.py port #sites opc
if len(sys.argv) > 1:
    port =  int(sys.argv[1])

if len(sys.argv) > 2:
    NoSites =  int(sys.argv[2])

if len(sys.argv) > 3:
    opc =  sys.argv[3]

STEPS = {"First":"Second","Third":"Fourth", "Fifth":"Sixth"}

#Three main tasks: 1.Initialize, 2.Move data to sites, 3.Interact with all sites, 4.Eval central collab model, 5.Collect all
def main():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind("tcp://*:%s" % port)

    sites = {}
    print("MainController listening all sites:", NoSites)
    time.sleep(30)

    #split data to be sended to all sites
    if opc == "sp":
        #0.'Site_1_of_3.csv': site1.ini
        print("MainController is generating data for all sites!!!")
        dataToCopy,remDir, shareDir = initialize.main(local_ip,port,NoSites)
    elif opc == "em":
        #Enable this option if maincrontroller must evaluate aggregated dataset
        print("MainController is ensembling data and models for all sites!!!")
        main_FDC.generateAllDataSets(file)
        main_FDC.ensembleModels(file)
        time.sleep(10)
    elif opc == "bc":
        while True:
            print("*******"*20)
            print("MainController is managing the construction of local model from all sites:", NoSites)
            j = sock.recv_json()
            step, site = j['msg'].split(",")
            sites[site]="First"

            if step == "First":
                print("First message received from Site:", site)
                send_next_step(sock,{"Central":"Second"})
                if valSites(sites,step):
                    print("sites",sites,"step",step)
                    break
            else:
                send_next_step(sock,{})

        time.sleep(2)
        while True:
            j = sock.recv_json()
            step, site = j['msg'].split(",")
            sites[site]="Third"
            if step == "Third":
                sock.send(b"STOP")
                if valSites(sites,step):
                    print("sites",sites,"step",step)
                    break
            else:
                send_next_step(sock,{})
    elif opc == "ac":
        print("*******"*20)
        print("MainController is managing the evaluation of collaborative model from all sites:", NoSites)
        while True:
            j = sock.recv_json()
            step, site = j['msg'].split(",")
            sites[site]="Third"

            if step == "Third":
                print("Third message received from Site:", site)
                send_next_step(sock,{"Central":"Fourth"})
                if valSites(sites,step):
                    print("sites",sites,"step",step)
                    break
            else:
                send_next_step(sock,{})

        time.sleep(2)
        while True:
            j = sock.recv_json()
            step, site = j['msg'].split(",")
            sites[site]="Fifth"
            if step == "Fifth":
                sock.send(b"STOP")
                if valSites(sites,step):
                    print("sites",sites,"step",step)
                    break
            else:
                send_next_step(sock,{})

    print("The main mainController has processed all sites:", str(NoSites))
    print("Successful process!!!")

    #Evaluate collaborative model
    if opc == "ev":
        main_FDC.evalCollabModels(file,"MC")
        print("The collaborative models have been satisfactorily evaluated.", str(NoSites))
        print("Successful process!!!")


def mainZMQ():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    if opc == "sp":
        #0.'Site_1_of_3.csv': site1.ini
        print("MainController is generating data for all sites!!!")
        dataToCopy,remDir, shareDir = initialize.main(local_ip,port,NoSites)
    else:
        #3.Execute steps in all Sites
        ExecStepsInAll()

    #4.Pending Eval model in central way


#From sites: 1st, 3rd, 5th, and Since MainController: 2nd, 4th
def ExecStepsInAll():
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind("tcp://*:%s" % port)

    sites = {}
    print("MainController listening all sites:", NoSites)
    time.sleep(2)

    for ss in STEPS:
        print("*******"*10)
        print("Processing Step:",ss)
        sendStep(sock,ss)

        if ss=="Third":
            #After receive first Messages
            print("\nGenerating testing data sets to all!!!")
            time.sleep(10)

            #Enable this option if maincrontroller must evaluate aggregated dataset
            main_FDC.generateAllDataSets(file)
            main_FDC.ensembleModels(file)
            time.sleep(10)

    print("The main mainController has processed all sites:", str(NoSites))
    print("Successful distributed process!!!")

def sendStep(sock,lblStep):
    sites = {}
    while True:
        msg = sock.recv_json()
        step, site = msg['msg'].split(",")
        sites[site]=lblStep

        if step == lblStep:
            print(step, "message received from Site:", site)
            if step == "Fifth":
                sock.send(b"STOP")
            else:
                send_next_step(sock,{"Central":STEPS[lblStep]})

            if valSites(sites,step):
                print("sites",sites,"step",step)
                break


def send_next_step(sock, step):
    try:
        sock.send_json(step)
    except StopIteration:
        sock.send_json({})


def valSites(sites, step):
    newDict = dict(filter(lambda elem: elem[1] == step, sites.items()))
    if len(newDict) == NoSites:
        print("\nMessages received from all Sites:", str(NoSites), ",Step:", step)
        return True
    return False

main()
