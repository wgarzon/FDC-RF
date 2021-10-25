import configparser
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
from shutil import copyfile
import shutil
import gdown
import time

#read configuration file with parameters for analysis process
def read_config_file(file='mainc.ini'):
    print("\n...Reading Config File!!!")
    config = configparser.ConfigParser()
    config.read(file)
    info, sections = dict(), config.sections()
    for section in sections:
        for val in config[section]:
            info[val] = config[section][val]

    return info


def set_ini_file(file_path, section, key, value):
    config = configparser.RawConfigParser()
    config.read(file_path)
    config.set(section,key,value)
    cfgfile = open(file_path,'w')
    config.write(cfgfile, space_around_delimiters=False)
    cfgfile.close()


def readData(name):
    df = pd.read_csv(name)
    print("Original dimension:",df.shape)
    df = df.rename(columns={'Unnamed: 0': 'sample_id'})
    return df


def get_chunks(list_, n):
    return [list_[start::n] for start in range(n)]


def splitUnBalData(path,file,dest,nsites):
    df = readData(path+file)
    files = {}

    s = list(range(df.shape[0]))
    random.shuffle(s)
    chunks = get_chunks(s, nsites)

    rta = []
    for listt in chunks: rta.append(df.iloc[listt,:])

    f = open(dest+ "/Split/class_distrib.txt", 'a')
    f.write("*"*10+ str(datetime.now().strftime("%d%m%Y_%H%M")) + "\n")

    for val in range(len(rta)):
        df = rta[val]
        name = "Site_" + str(val+1) +"_of_" + str(nsites) +".csv"
        files[name] = dest
        print("Dimension Site" + str(val+1) + ":"+ str(df.shape))
        #print(df.groupby(['Class']).size())
        df.to_csv(dest+"/Split/"+name,index=False)
        f.write("Site " + str(val+1) + ":\n")
        if 'Class' in df.columns:
            f.write(str(df.groupby(['Class']).size())+"\n")
        else:
            f.write(str(df.groupby(['ClassNum']).size())+"\n")

    f.close()
    print(str(nsites) + "Files Split!!!:")
    return files


#get data from google drive
def getGDriveData(id,file,path):
    url = 'https://drive.google.com/uc?id=' + id
    gdown.download(url, file, quiet=False)
    time.sleep(5)
    shutil.move(file, path+file)
    return True


def main(ip,port,nsites):
    sections = read_config_file()
    dataDir = sections["data_dir"]
    shareDir = sections["share_models"]
    remDir = sections["remote_folder"]

    if not os.path.exists(dataDir): os.makedirs(dataDir)
    if not os.path.exists(shareDir): os.makedirs(shareDir)
    if not os.path.exists(shareDir+ "/Split"): os.makedirs(shareDir + "/Split")

    dataFileName = sections["ds_name"]
    fromRepo = sections["fromrepo"]
    #download data from link provided
    if fromRepo =="Yes":
        if not getGDriveData(sections["linkid"],dataFileName,dataDir):
            raise "Error downloading the file " +dataFileName

    time.sleep(10)
    files = splitUnBalData(dataDir,dataFileName,shareDir,nsites)

    #'Site_1_of_3.csv': '~/Remote_FDC_RF/data/'
    file = "site.ini"
    currDir = os.getcwd() + "/" + file

    #replicate .ini files to be copy to other Sites
    k = 1
    for ff in files:
        newFile = file.replace(".ini",str(k)+".ini")
        files[ff] = newFile

        newFile = shareDir + "/Split/" + newFile
        copyfile(currDir, newFile)
        set_ini_file(newFile,"Paths","ds_name", ff)
        set_ini_file(newFile,"Site","id", str(k))
        set_ini_file(newFile,"Site","ip", ip)
        set_ini_file(newFile,"Site","port", str(port))
        k+=1

    return files, remDir, shareDir
