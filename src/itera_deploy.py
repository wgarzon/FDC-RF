from execo import *
from execo_g5k import *
import time
import sys
import math
import shutil
import os

remFolder = "~/Remote_FDC_RF/"
dirLocResults = "/home/wgarzon/getResults/"
init_cmd_all =  """mkdir -p ~/Remote_FDC_RF/"""
init_cmd_site = """mkdir -p ~/Remote_FDC_RF/data/"""
init_sites = """rm -rf ~/Remote_FDC_RF/"""
putKey = """scp -o BatchMode=yes -o PasswordAuthentication=no -o StrictHostKeyChecking=no ./.ssh/id_rsa %s:/root/.ssh/"""
scpCMD = "scp -o BatchMode=yes -o PasswordAuthentication=no -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -rp -o User=root"

#python3 mainControl.py port #sites opc
InvokeMC = """
            cd ~/Remote_FDC_RF/
            python3 mainControl.py %s %d %s
            """
#python3 siteControl.py port siteid ip
InvokeSC = """
            cd ~/Remote_FDC_RF/
            python3 siteControl.py %s"""

commFiles = ["aggregator.py","FDC_RF.py","main_FDC.py","utils.py","bestParams.py"]
mainCFiles = ["initialize.py","mainControl.py","mainc.ini","site.ini","data"]
siteCFiles = ["siteControl.py"]
centralFiles = ["central"]


def get_nodes(jobid,site):
    return get_oar_job_nodes(jobid, site)

def sub_deploy(nodes_number,site,gksites):
    [(jobid,site)] = oarsub([(OarSubmission(resources="{cluster='" + gksites[site] + "'}/nodes="+str(nodes_number), job_type='deploy'), site)])
    #res = oarsub([(OarSubmission(resources=['{cluster=\'paravance\'}/nodes=3', '{type=\'kavlan-local\'}/vlan=1'], walltime='1:00:00', job_type='deploy’), ‘rennes’)])
    assert jobid, "Job has not been created!"
    print(jobid)

    wait_oar_job_start(jobid, site)
    print("started")
    nodes = get_oar_job_nodes(jobid, site)
    my_deploy = Deployment(hosts=nodes, env_name='ubuntu1804-x64-min', other_options='-k')
    deployed, undeployed = deploy(deployment = my_deploy)
    assert len(undeployed) == 0, "Deploying failed for %d node(s)" % len(undeployed)
    return nodes

def flatten(list):
    flat_list = []
    for sublist in list: flat_list.append(sublist)

    return flat_list

def exec_cmd(cmd, node, **kwargs):
    nodes = node
    if isinstance(node, list):
        proc = Remote(cmd, node, **kwargs)
        proc.run()
        stdout = [p.stdout for p in proc.processes]
    else:
        proc = SshProcess(cmd, node, **kwargs)
        proc.run()
        stdout = proc.stdout
    return proc.ok, stdout

def exec_cmd_as_root(cmd, node):
    return exec_cmd(cmd, node, connection_params={'user': 'root'})

#Always sort the nodes to define the lower as MainC
def FDC_Hosts(nodes,centralSite):
    first = [c.address for c in nodes]
    first.sort()
    allSites = []

    for nn in first:
        for node in nodes:
            if node.address == nn:
                allSites.append(node)

    if centralSite:
        return allSites[0], allSites[1], allSites[2:]
    else:
        return "", allSites[0], allSites[1:]

def mv_data_to_sites(sites, mainC):
    for i in range(len(sites)):
        source = "Site_X_of_Y.csv".replace("Y",str(len(sites))).replace("X",str(i+1))
        cmd = scpCMD + " ~/Remote_FDC_RF/shareModels/Split/"+source+ " root@"+sites[i].address+ ":~/Remote_FDC_RF/data/"+source
        add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
        add_resources.run()
        #print("stdout",add_resources.stdout)
        sleep(2)

def mv_ini_to_sites(sites, mainC):
    for i in range(len(sites)):
        source = "site.ini".replace(".ini",str(i+1)+".ini")
        cmd = scpCMD + " ~/Remote_FDC_RF/shareModels/Split/"+source+ " root@"+sites[i].address+ ":~/Remote_FDC_RF/"+source
        add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
        add_resources.run()
        #print("stdout",add_resources.stdout)
        sleep(2)

def DistributedProcess(allSites,mainC,port,nodes_number, step):
    actions = []
    cmd = InvokeSC % (step)
    for val in range(len(allSites)):  actions.append(Remote(cmd,allSites[val].address,connection_params={'user': 'root'}))
    cmd = InvokeMC % (port,nodes_number,step)
    actions.append(Remote(cmd,mainC.address,connection_params={'user': 'root'}))

    para = ParallelActions(actions)
    para.run()
    #print("DistributedProcess",para)

    #for rem in actions:
    #    stdout = [p.stdout for p in rem.processes]
    #    print(stdout)


def mv_data_to_mainc(sites, mainC):
    for i in range(len(sites)):
        cmd = scpCMD + " root@"+sites[i].address+ ":~/Remote_FDC_RF/shareModels/ ~/Remote_FDC_RF/"
        add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
        add_resources.run()
        print("mv_data_to_mainc",add_resources.stdout)
        sleep(2)

def mv_results_to_mainc(sites, mainC):
    for i in range(len(sites)):
        cmd = scpCMD + " root@"+sites[i].address+":~/Remote_FDC_RF/shareModels/outcomes/ ~/Remote_FDC_RF/shareModels/"
        add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
        add_resources.run()
        print("mv_results_to_mainc",add_resources.stdout)
        sleep(2)

def mv_model_to_sites(sites, mainC):
    for i in range(len(sites)):
        cmd = scpCMD + " ~/Remote_FDC_RF/shareModels/Ensembled/ root@"+sites[i].address+":~/Remote_FDC_RF/shareModels/"
        add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
        add_resources.run()
        print("mv_model_to_sites",add_resources.stdout)
        sleep(2)

def mv_model_to_central(central, mainC):
    cmd = scpCMD + " ~/Remote_FDC_RF/shareModels/Ensembled/ root@"+central.address+":~/Remote_FDC_RF/central/shareModels/"
    add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
    add_resources.run()
    print("mv_model_to_central Node",add_resources.stdout)
    sleep(2)

def mv_data_to_central(central, mainC):
    cmd = scpCMD + " ~/Remote_FDC_RF/data/ root@"+central.address+ ":~/Remote_FDC_RF/central/data/"
    add_resources = SshProcess(cmd, mainC.address, connection_params={'user': 'root'})
    add_resources.run()
    sleep(2)

def installLibraries(nodes):
    exec_cmd_as_root("apt-get update && apt-get install python3-pip -y", nodes)
    exec_cmd_as_root("pip3 install Cython -U scikit-learn==0.22 pandas matplotlib imblearn && pip3 install pyzmq && pip3 install gdown", nodes)


#python3 itera_deploy.py 5555 3 1 10
def main():
    job_id = 2864835
    deploy = True
    instLibs = True
    initSites = True
    nodes_number = 3
    port = "5557"
    site = 'nantes'

    #gksites = {"nancy":"gros,grisou,grimoire","nantes":"econome", "rennes":"parasilo"}

    gksites = {"nancy":"grisou","nantes":"econome", "rennes":"parasilo"}
    if len(sys.argv) > 1: port =  int(sys.argv[1])
    if len(sys.argv) > 2: nodes_number = int(sys.argv[2])+1
    if len(sys.argv) > 3: ffrom =  int(sys.argv[3])
    if len(sys.argv) > 4: to =  int(sys.argv[4])

    nodes = []

    if deploy:
        nodes = sub_deploy(nodes_number,site,gksites)
        #Install remote libraries in all nodes
        Put(nodes, ["~/.ssh/id_rsa","~/.ssh/id_rsa.pub"], "/root/.ssh/", connection_params = {'user': 'root'}).run()
        installLibraries(nodes)
    else:
        nodes = get_nodes(job_id,site)
        print("======= View reservations information =======")
        infos = get_oar_job_info(job_id, site)
        for info in infos: print(" * %s --> %s" % (info, infos[info]))

    Put(nodes, ["~/.ssh/id_rsa","~/.ssh/id_rsa.pub"], "/root/.ssh/", connection_params = {'user': 'root'}).run()

    addCentral = False
    allNodes = flatten(nodes)
    if instLibs: installLibraries(allNodes)

    print("allNodes",allNodes)
    centralNode, mainCsite, allSites = FDC_Hosts(allNodes,addCentral)

    if initSites:
        ok, stdout = exec_cmd_as_root(init_sites, allNodes)
        print("stdout",stdout)

    for ttime in range(ffrom,to+1):
        #ok, stdout = exec_cmd_as_root(init_sites, nodes)
        #Init folder in all sites
        print("*** Processing time", ttime)
        ok, stdout = exec_cmd_as_root(init_cmd_all, allNodes)
        ok, stdout = exec_cmd_as_root(init_cmd_site, allSites)

        print("Main Controller Node:", mainCsite)
        print("Site Nodes:", allSites)

        #run initialization tasks only the first time
        if ttime==ffrom:
            #Copy commFiles in all nodes
            Put(nodes, commFiles, remFolder, connection_params = {'user': 'root'}).run()
            print("ok: Copying common files in all sites")

            #Copy Files in MainController site
            Put(mainCsite, mainCFiles, remFolder, connection_params = {'user': 'root'}).run()
            print("ok: Copying files into mainController site")

            #Copy Files in SiteController site
            Put(allSites, siteCFiles, remFolder, connection_params = {'user': 'root'}).run()
            print("ok: Copying files in all SiteControllers")

        #split data to be analyzed
        ok, stdout = exec_cmd_as_root(InvokeMC % (port,len(allSites),"sp"),mainCsite)
        print("ok: Data partitioned correctly!!!")
        print("stdout",stdout)

        #move CSV data to all sites
        mv_data_to_sites(allSites, mainCsite)
        print("ok: Main Controller copy CSV files to all sites")

        #move INI files to all sites
        mv_ini_to_sites(allSites, mainCsite)
        print("ok: Main Controller copy INI files to all sites")

        #Start firsts steps of the process between sites and mainController
        DistributedProcess(allSites,mainCsite,port,len(allSites),"bc")
        print("ok: First steps of the distributed process executed correctly")

        #get data from all sites
        mv_data_to_mainc(allSites, mainCsite)
        print("ok: Main Controller got all data from sites")

        #ensemble models and aggregated data
        ok, stdout = exec_cmd_as_root(InvokeMC % (port,len(allSites),"em"),mainCsite)
        print("ok: Main Controller ensembled models and data for all sites")
        print("stdout",stdout)

        mv_model_to_sites(allSites, mainCsite)
        print("ok: Main Controller move models to all sites")


        #Start last steps of the process between sites and mainController
        DistributedProcess(allSites,mainCsite,port,len(allSites),"ac")
        print("ok: The local and collaborative models on each site has been evaluated!!!")

        #Collect results from all sites
        mv_results_to_mainc(allSites, mainCsite)
        print("ok: Main Controller get all data from sites")

        #ensemble models and aggregated data
        ok, stdout = exec_cmd_as_root(InvokeMC % (port,len(allSites),"ev"),mainCsite)
        print("ok: Main Controller evaluated collaborative models using testing aggregated dataset")
        print("stdout",stdout)

        #get the results from remote sites to local folder
        Get(mainCsite, ["~/Remote_FDC_RF/shareModels/outcomes/"], dirLocResults, connection_params = {'user': 'root'}).run()
        Get(mainCsite, ["~/Remote_FDC_RF/shareModels/Ensembled/"], dirLocResults, connection_params = {'user': 'root'}).run()

        #create the folder with the number of iteration
        if not os.path.exists(dirLocResults+str(ttime)): os.makedirs(dirLocResults+str(ttime)+"/")
        shutil.move(dirLocResults+"outcomes/", dirLocResults+str(ttime))
        shutil.move(dirLocResults+"Ensembled/", dirLocResults+str(ttime))

main()
