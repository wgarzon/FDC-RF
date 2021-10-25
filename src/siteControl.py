import zmq
import sys
import time
from utils import *
from aggregator import *
import main_FDC
from copy import deepcopy
from datetime import datetime
import os
import os.path, time

opc = ""

if len(sys.argv) > 1:
    opc =  sys.argv[1]

#get ini file after 30sec of creation
def getIniFile():
    ffiles = []
    while len(ffiles)==0: ffiles = [i for i in os.listdir(os.getcwd()) if i.endswith(".ini")]
    file = ffiles[0]
    t1 = datetime.strptime(time.ctime(os.path.getctime(file)), "%c")
    while (datetime.now()-t1).total_seconds()<5: t1 = datetime.strptime(time.ctime(os.path.getctime(file)), "%c")
    return file

def main():
    config = getIniFile()

    #get information of ip, id and port
    rta = read_config_file(config)
    site =  int(rta["id"])
    port =  int(rta["port"])
    ip =  rta["ip"]

    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://%s:%s" % (ip,port))

    print("Site " + str(site) + " is waiting message from the MainController")

    #Before Collaborative Model
    if opc=="bc":
        while True:
            sock.send_json({ "msg": "First,"+str(site)})
            work = sock.recv_json()
            if work != {}: print("First message sent!!!")
            if work == {}: continue

            step = work['Central']
            print("\nMessage received from Central Site, Step:%s" % (step))

            #build the function to build the local model
            buildLocalModel(config, site)
            time.sleep(5)

            sock.send_json({ "msg": "Third,"+str(site)})
            rta = sock.recv()
            if rta == b"STOP": break
    else: ##After Collaborative Model
        while True:
            #Step3: send a message after build the local model
            sock.send_json({ "msg": "Third,"+str(site)})
            work = sock.recv_json()
            if work != {}: print("Third message sent!!!")
            if work == {}: continue
            step = work['Central']
            print("\nMessage received from Central Site, Step:%s" % (step))

            print("Evaluating collaborative model over local data")
            EvalCollabModel=False
            main_FDC.TestModels(config, str(site), True, EvalCollabModel, False)
            time.sleep(5)

            #Step5: send a message after evaluete collaborative model
            sock.send_json({ "msg": "Fifth,"+str(site)})
            print("Fifth message sent!!!")
            rta = sock.recv()
            if rta == b"STOP": break

def mainZMQ():
    ffiles = [i for i in os.listdir(os.getcwd()) if i.endswith(".ini")]
    config = ffiles[0]

    #get information of ip, id and port
    rta = read_config_file(config)
    site =  int(rta["id"])
    port =  int(rta["port"])
    ip =  rta["ip"]

    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://%s:%s" % (ip,port))

    print("Site " + str(site) + " is waiting message from the MainController")

    while True:
        sock.send_json({ "msg": "First,"+str(site)})
        work = sock.recv_json()
        if work != {}: print("First message sent!!!")
        if work == {}: continue
        step = work['Central']
        print("\nMessage received from Central Site, Step:%s" % (step))

        #build the function to build the local model
        buildLocalModel(config, site)
        time.sleep(5)

        #Step3: send a message after build the local model
        sock.send_json({ "msg": "Third,"+str(site)})
        work = sock.recv_json()
        if work != {}: print("Second message sent!!!")
        if work == {}: continue
        step = work['Central']
        print("\nMessage received from Central Site, Step:%s" % (step))

        print("Evaluating collaborative model")
        main_FDC.TestModels(config, str(site), True, True, True)
        time.sleep(5)

        #Step5: send a message after evaluete collaborative model
        sock.send_json({ "msg": "Fifth,"+str(site)})
        print("Fifth message sent!!!")
        rta = sock.recv()
        if rta == b"STOP": break

    print("\nThe site", str(site), "has finished the process!!!")


def buildLocalModel(config, site):
    print("Building Local Model, Site:",site)
    #Generate local models
    #t1 = datetime.now()
    Xtest, ytest, Xtrain, localMdl, delta = main_FDC.localModel(config, str(site))
    #t2 = datetime.now()
    #delta = t2 - t1
    f = open("shareModels/outcomes/Training_Time" + str(site) +".txt", "a")
    f.write("Site" + str(site) + "," + str(delta.total_seconds()) + "\n")
    f.close()

main()
