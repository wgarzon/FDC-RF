from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import configparser
import numpy as np
import random
from joblib import dump
from sklearn import tree
import os
from datetime import datetime
import matplotlib.pyplot as plt
import math

#read configuration file with parameters for analysis process
def read_config_file(file='config.ini'):
    print("\n...Reading Config File!!!")
    config = configparser.ConfigParser()
    config.read(file)
    info, sections = dict(), config.sections()
    for section in sections:
        for val in config[section]:
            info[val] = config[section][val]

    return info


def show_info_ds(data):
    print("Shape:\n", data.shape)
    print("Head:\n", data.head())
    print("Group by:\n", data.groupby('Class').size())


def preprocess_ds(dsName, pathShare, site, norm=True, genFiles=False):
    df = pd.read_csv(dsName)
    dataSet = df.dropna()
    print("Reading Data Set!!!\nNumber of Samples:" + str(dataSet.shape[0]) + ", Features:" + str(dataSet.shape[1]))
    data, DSnonClass, X, y = prepare_ds(dataSet)
    # Normalize original DS
    if norm: X = normalize_ds(X)

    # some changes to reduce the accuracy
    # addNoise(X)
    # X = addNoise2(X, 0.7)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    # Generate CSV files with data used during the analysis process
    if genFiles:
        testDS = pd.concat([Xtest, ytest], axis=1)
        testDS.to_csv(pathShare+"Site"+str(site)+"_testDS.csv", index=False)
        #ytrain.to_csv(pathShare+"Site"+str(site)+"_ytrain.csv", index=False)
        #ytest.to_csv(pathShare+ "Site"+str(site)+"_ytest.csv", index=False)
        #pd.DataFrame(Xtest).to_csv(pathShare+"Site"+str(site)+"_xtest.csv", index=False)
        #pd.DataFrame(Xtrain).to_csv(pathShare + "Site"+str(site)+"_xtrain.csv", index=False)

    return Xtrain, Xtest, ytrain, ytest


def loadAggData(pathOut):
    testDS = pd.read_csv(pathOut+"testDS_Agg.csv")
    Xtest = testDS.drop(['ClassNum'], axis=1)
    ytest= testDS['ClassNum']

    return Xtest, ytest


# Prepare DS excluding class and features variables from data
def prepare_ds(data):
    clases = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']
    dicti = dict(zip(clases, [i for i in range(len(clases))]))
    # data['Class'].map({'BRCA': 0, 'COAD': 1, 'KIRC': 2, 'LUAD': 3, 'PRAD': 4}).astype(int)
    # map each value class with an integer value since zero
    # remove classes from dataframe
    if 'Class' in data.columns:
        data['ClassNum'] = data['Class'].map(dicti).astype(int)
        DSnonClass = data.drop(['sample_id', 'Class'], axis=1)
    else:
        DSnonClass = data.copy()

    X = DSnonClass.drop(['ClassNum'], axis=1)
    Y = DSnonClass['ClassNum']

    return data, DSnonClass, X, Y


# Nomalize data set using MinMaxScaler method
def normalize_ds(data):
    x = data.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    return df


def addNoise(df):
    mu, sigma = 0, 0.5
    # creating a noise with the same dimension as the dataset DF
    noise = np.random.normal(mu, sigma, df.shape)
    noiseDF = df + noise


def addNoise2(ndf, perc):
    df = ndf.copy()
    rows = df.shape[0]
    cols = df.shape[1]

    randIDRows = random.sample(range(0, rows), int(perc * rows))
    randIDCols = random.sample(range(0, cols), int(perc * cols))

    df.iloc[randIDRows, randIDCols] = ndf.iloc[randIDRows, randIDCols] + random.uniform(-1, 10)
    return df


def saveMdlDisk(mdl, path, pathShare, site):
    if path != "": dump(mdl, path + site)
    dump(mdl, pathShare + site)
    print("Local Model stored on disk!!!")

def saveTree(no, tt, site):
    return True
    f = site + "_Model" + str(no)+ ".dot"
    with open(f, 'w') as my_file:
        my_file = tree.export_graphviz(tt, out_file=my_file, impurity = True, class_names = ['1', '2', '3', '4', '5'], rounded = True, filled= True )

    #by shell
    #dot -Tpng _Model1.dot -o _Model1.png
    #print_decision_tree(tt)

#get list of features importance variables
def save_VI_file(model, df, site , path,idd=0):
    fi = pd.DataFrame({'cols':df.columns, 'imp':model.feature_importances_}).sort_values('imp', ascending=False)
    res = fi[:15]
    res['Version'] = [site]*15
    res['ID'] = [idd]*15
    head = False
    if not os.path.exists(path+'outcomes/VIM.csv'): head = True
    res.to_csv(path+'outcomes/VIM_'+site+'.csv', mode='a',  index=False, header=head)
    plot_VIM(fi[:30],site,path)

#Plot features by level importance
def plot_VIM(df,site,path):
    df.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

    plt.xlabel("Variable Importance Score")
    plt.ylabel("Genes ID")
    plt.title("Relevant Genes - "+site)
    #plt.show()
    xx = datetime.now().strftime("_%d%m%Y_%H%M")
    plt.savefig(path + "outcomes/VIM_" + site+xx+'.png')


#Plot all OOB data for each decision tree
def plot_OOB(d,y_label,site, outDir):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    fig, ax = plt.subplots()
    ax.plot(x, y)
    step = math.ceil(max(x))//10
    xint = range(min(x), math.ceil(max(x))+5,step)
    ax.set_xticks(xint)
    ax.set_xlabel("Decision Tree ID")
    ax.set_ylabel(y_label)
    ax.set_title(y_label + ' for each DT - ' + site)

    xx = datetime.now().strftime("_%d%m%Y_%H%M")
    fig.savefig(outDir + 'outcomes/OOB_'+site+xx+'.png')


def writeLog(path,site,txt):
    f = open(path+"outcomes/Log"+site+".txt", "a")
    f.write("\n"+txt)
    f.close()

#save models by precision accuracy
def saveMdlsByAcc(mdlsByAcc, site, path,pathShare, type):
    #dump(mdlsByAcc[0], path + site + "Model_Bad.Accmodel")
    dump(mdlsByAcc[1], path + site + "Model_Best."+type+"Accmodel")
    dump(mdlsByAcc[1], pathShare + site + "Model_Best."+type+"Accmodel")

    txt = type + "No of Best models:" + str(len(mdlsByAcc[1])) + ". No of Bad models:" + str(len(mdlsByAcc[0]))
    print(txt)
    writeLog(pathShare,site,txt)
    print("Share Models stored on disk!!!")


def aggDataSets(path):
    dfs = []
    allffs = [path + i for i in os.listdir(path) if i.endswith("_testDS.csv")]
    for ff in allffs: dfs.append(pd.read_csv(ff))
    res = pd.concat(dfs)
    res.to_csv(path +"Ensembled/testDS_Agg.csv", index=False)

    print("All testing files generated to be shared with all sites")
    #return files["xtrain"]
