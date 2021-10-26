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

import matplotlib.pyplot as plt
import numpy as np

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)



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


#adjusted to demo version
def LoadData(site, dsName, shareMDir, file=""):
    print("***** Processing " + site + " *****")
    if file=="":
        Xtrain, Xtest, ytrain, ytest = preprocess_ds(dsName, shareMDir, site, True, True)
    else:
        Xtrain, Xtest, ytrain, ytest = preprocess_ds(file, shareMDir, site, True, True)

    return Xtrain, Xtest, ytrain, ytest

#adjusted to demo version
def preprocess_ds(dsName, pathShare, site, norm=True, genFiles=False):
    df = pd.read_csv(dsName, engine='python')
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
        pd.DataFrame(Xtrain).to_csv(pathShare + "Site"+str(site)+"_xtrain.csv", index=False)

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
