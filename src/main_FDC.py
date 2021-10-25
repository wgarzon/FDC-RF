import FDC_RF as fdc
from utils import *
from aggregator import *
import warnings
from joblib import load
import os
import os.path, time

warnings.filterwarnings("ignore")
global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts

def initializeEnv(file):
    #read config values at site
    confValues = read_config_file(file)
    global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts, dirFigs
    if file == "mainc.ini":
        shareMDir = confValues['share_models']
        if not os.path.exists(shareMDir): os.makedirs(shareMDir)
        if not os.path.exists(shareMDir+"Ensembled"): os.makedirs(shareMDir+"Ensembled")
        return

    dsName, outDir, site, DTs, shareMDir = confValues['ds_dir'] + confValues['ds_name'], confValues['out_dir'], confValues['id'], int(confValues['ntrees']), confValues['share_models']
    mdlName, best_dts = confValues['model_name'], float(confValues['best_dts'])

    if not os.path.exists(outDir): os.makedirs(outDir)
    if not os.path.exists(shareMDir + "outcomes"): os.makedirs(shareMDir + "outcomes")
    if not os.path.exists(shareMDir): os.makedirs(shareMDir)


def genLocalModel(Xtrain, ytrain):
    print("... Generating Local Model")

    #build model according to param file
    t1 = datetime.now()
    fdcRF = fdc.buildModel(mdlName,DTs,Xtrain,ytrain)
    t2 = datetime.now()
    delta = t2 - t1
    fdcMdl = fdcRF.getModel()

    #Get details about the model into dictionaries structure to be merged
    #infoM1, infoM2, maxFeat = fdc.getInfoDTs(fdcMdl, fdcRF.nFeats)
    # Print information about DTs and stored them into .dot file
    #fdc.ShowInfoDTs(fdcMdl, site)
    #Parallelizable method to compute OOB Error
    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)
    #select best p decision trees
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    txt = "Threshold used:" + str(thresh)
    print(txt)
    writeLog(shareMDir,"Site"+site,txt)

    mdlsByAcc = fdc.splitMdlsByAcc(bestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    #mdlsByAcc, featsByAcc = fdc.splitMdlsByAcc(bestDTs, infoM1,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, "Site"+site, outDir, shareMDir, "Perc")

    #Get the best models from median value
    nbestDTs, median = fdc.getBestByMedian(OOB_Acc)
    txt = "Threshold used MEDIAN:" + str(median)
    print(txt)
    writeLog(shareMDir,"Site"+site,txt)

    #mdlsByAcc, featsByAcc = fdc.splitMdlsByAcc(bestDTs, infoM1,[Xtrain,ytrain],mdlName)
    mdlsByAcc = fdc.splitMdlsByAcc(nbestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, "Site"+site, outDir, shareMDir, "Medn")

    #print("ModelsByAccuracy",mdlsByAcc)
    saveMdlDisk(fdcMdl,outDir,shareMDir, "Site"+site+ "Complete.smodel")

    return mdlsByAcc, fdcMdl, delta


def getLocalModel():
    localmodel = loadLocalModel(shareMDir, site, "smodel")
    return localmodel


#Build the local model
def localModel(file, site):
    initializeEnv(file)
    print("***** Processing " + site + " *****")
    Xtrain, Xtest, ytrain, ytest = preprocess_ds(dsName, shareMDir, site, True, True)
    mdlsByAcc, fdcMdl, delta = genLocalModel(Xtrain, ytrain)
    return Xtest, ytest, Xtrain, fdcMdl, delta

#join all datasets into one CSV to evaluate the models
def generateAllDataSets(file):
    initializeEnv(file)
    aggDataSets(shareMDir)

#function to evaluate collaborative models using aggdata
def evalCollabModels(file,site):
    initializeEnv(file)
    nPath = shareMDir+"Ensembled/"

    Xtest, ytest = loadAggData(shareMDir+"Ensembled/")
    dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)

    print("Testing ", dim)
    nPath = shareMDir+"Ensembled/"
    print("... Evaluating Join and Best Collaborative Model")
    globalModel = load(nPath+"JoinCollab.jcmodel")
    fdc.ScoreModel("Join" + site,site,shareMDir, globalModel, Xtest, ytest,dim)

    BestGlobModel = load(nPath+"BestCollab.pbcmodel")
    fdc.ScoreModel("BestPerc"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)

    BestGlobModel = load(nPath+"BestCollab.mbcmodel")
    fdc.ScoreModel("BestMedn"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)


#ensemble the models per site into two collaborative models
def ensembleModels(file):
    initializeEnv(file)
    nPath = shareMDir+"Ensembled/"
    print("... Generating Joining Model")
    globalModel = mergeModelsJoin(shareMDir, "smodel")
    saveMdlDisk(globalModel, "", nPath, "JoinCollab.jcmodel")

    print("... Generating Best Models")
    BestGlobModel = mergeModelsAcc(shareMDir, "PercAccmodel")
    saveMdlDisk(BestGlobModel, "", nPath, "BestCollab.pbcmodel")

    BestGlobModel = mergeModelsAcc(shareMDir, "MednAccmodel")
    saveMdlDisk(BestGlobModel, "", nPath, "BestCollab.mbcmodel")
    print("Two models generated Successful!!!")


#evaluate the local and collaborative model (optional)
def TestModels(file, site, lcMdl=True, colMdl=False, VIM=False):
    initializeEnv(file)

    if lcMdl:
        Xtest, ytest = loadAggData(shareMDir+"Ensembled/")
        dim = "Local Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)
        print("Testing ", dim)
        nn = "SiteXComplete.smodel".replace("X",str(site))
        localModel = load(outDir+nn)
        fdc.ScoreModel("Site"+str(site),str(site), shareMDir,localModel, Xtest, ytest, dim)
        if VIM: save_VI_file(localModel, Xtrain, "Local"+site, shareMDir)

    if colMdl:
        Xtest, ytest = loadAggData(shareMDir+"Ensembled/")
        dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)
        print("Testing ", dim)
        nPath = shareMDir+"Ensembled/"
        print("... Evaluating Join and Best Collaborative Model")
        globalModel = load(nPath+"JoinCollab.jcmodel")
        fdc.ScoreModel("Join" + site,site,shareMDir, globalModel, Xtest, ytest,dim)

        BestGlobModel = load(nPath+"BestCollab.pbcmodel")
        fdc.ScoreModel("BestPerc"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)

        BestGlobModel = load(nPath+"BestCollab.mbcmodel")
        fdc.ScoreModel("BestMedn"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim)

def test():
    localModel("site.ini", "1")

#test()
