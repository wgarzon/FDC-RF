#main_FDC version to execute in local scenario
import FDC_RF as fdc
from utils import *
from aggregator import *
import warnings
from joblib import load
import os
import os.path, time

warnings.filterwarnings("ignore")
global dsName, outDir, site, DTs, shareMDir, mdlName, best_dts

#method adjusted to execute in local version
def initializeEnv(file=""):
    if file=="": file = [i for i in os.listdir(os.getcwd()) if i.endswith(".ini")][0]
    #read config values at site
    confValues = read_config_file(file)
    dictIni = {}

    dictIni["ds_dir"], dictIni["dsName"], dictIni["outDir"] = confValues['ds_dir'], confValues['ds_dir'] + confValues['ds_name'], confValues['out_dir']
    dictIni["site"], dictIni["DTs"], dictIni["shareMDir"] = int(confValues['id']), int(confValues['ntrees']), confValues['share_models']
    dictIni["mdlName"], dictIni["best_dts"] = confValues['model_name'], float(confValues['best_dts'])

    if not os.path.exists(dictIni["ds_dir"]): os.makedirs(dictIni["ds_dir"])
    if not os.path.exists(dictIni["shareMDir"] + "outcomes"): os.makedirs(dictIni["shareMDir"] + "outcomes")
    if not os.path.exists(dictIni["shareMDir"] + "Ensembled"): os.makedirs(dictIni["shareMDir"] + "Ensembled")
    if not os.path.exists(dictIni["shareMDir"]): os.makedirs(dictIni["shareMDir"])
    if not os.path.exists(dictIni["outDir"]): os.makedirs(dictIni["outDir"])

    return dictIni


#Method adjusted to train a central model
def getCentralModel(Xtrain, ytrain, dictIni,best_dts,site):
    mdlName,DTs,outDir,shareMDir = dictIni["mdlName"],dictIni["DTs"],dictIni["outDir"],dictIni["shareMDir"]
    print("... Generating Central Model")

    #build model according to param file
    t1 = datetime.now()
    fdcRF = fdc.buildModel(mdlName,DTs,Xtrain,ytrain)
    t2 = datetime.now()
    delta = t2 - t1
    fdcMdl = fdcRF.getModel()

    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)
    #get best decision trees by percentage
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    mdlsByAcc = fdc.splitMdlsByAcc(bestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, site, outDir, shareMDir, "Perc")
    #Get the best models from median value
    nbestDTs, median = fdc.getBestByMedian(OOB_Acc)
    mdlsByAcc = fdc.splitMdlsByAcc(nbestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, site, outDir, shareMDir, "Medn")
    saveMdlDisk(fdcMdl,outDir,shareMDir, "CentralModel.smodel")

    #return mdlsByAcc, fdcMdl, delta
    return fdcMdl, delta

#function adjusted to demo version
def genLocalModel(Xtrain, ytrain, dictIni,best_dts,site):
    mdlName,DTs,outDir,shareMDir = dictIni["mdlName"],dictIni["DTs"],dictIni["outDir"],dictIni["shareMDir"]
    print("... Generating Local Model")
    #build model according to param file
    t1 = datetime.now()
    fdcRF = fdc.buildModel(mdlName,DTs,Xtrain,ytrain)
    t2 = datetime.now()
    delta = t2 - t1
    fdcMdl = fdcRF.getModel()

    #Get details about the model into dictionaries structure to be merged
    OOB_Err, OOB_Acc, AggAcc = fdc.ParOOBErrorTree(fdcMdl, Xtrain, ytrain)
    #select best p decision trees
    bestDTs, thresh = fdc.getBestByPerc(OOB_Acc, best_dts)
    txt = "Threshold used:" + str(thresh)
    print(txt)
    writeLog(shareMDir,"Site"+site,txt)

    mdlsByAcc = fdc.splitMdlsByAcc(bestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, "Site"+site, outDir, shareMDir, "Perc")

    #Get the best models from median value
    nbestDTs, median = fdc.getBestByMedian(OOB_Acc)
    txt = "Threshold used MEDIAN:" + str(median)
    print(txt)
    writeLog(shareMDir,"Site"+site,txt)

    mdlsByAcc = fdc.splitMdlsByAcc(nbestDTs, fdcMdl,[Xtrain,ytrain],mdlName)
    saveMdlsByAcc(mdlsByAcc, "Site"+site, outDir, shareMDir, "Medn")
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

#function adjusted to demo Version
#to evaluate collaborative models using aggdata
def evalCollabModels(site, shareMDir, show=False):
    nPath = shareMDir+"Ensembled/"

    Xtest, ytest = loadAggData(shareMDir+"Ensembled/")
    dim = "Collab Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)

    print("Testing ", dim)
    nPath = shareMDir+"Ensembled/"
    print("... Evaluating Join and Best Collaborative Model")
    globalModel = load(nPath+"JoinCollab.jcmodel")
    _, _, _, reportJoin = fdc.ScoreModel("Join" + site,site,shareMDir, globalModel, Xtest, ytest,dim,show)

    BestGlobModel = load(nPath+"BestCollab.pbcmodel")
    _, _, _, reportBPerc = fdc.ScoreModel("BestPerc"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim,show)

    BestGlobModel = load(nPath+"BestCollab.mbcmodel")
    _, _, _, reportBMedn = fdc.ScoreModel("BestMedn"+site, site, shareMDir, BestGlobModel,Xtest,ytest,dim,show)

    return reportJoin, reportBPerc, reportBMedn

#Function adapted to demo version
def evaluateCentralModel(site, shareMDir, show=False):
    print("... Evaluating Central Model")

    Xtest, ytest = loadAggData(shareMDir+"Ensembled/")
    dim = "Central Model:"+",Xtest:"+str(Xtest.shape)+ ",ytest:"+ str(ytest.shape)
    print("Testing ", dim)
    CompleteModel = load(shareMDir+"CentralModel.smodel")
    _, _, _, reportJoin = fdc.ScoreModel("CentrModel", site, shareMDir, CompleteModel,Xtest,ytest,dim,show)

    BestCentralModel = load(shareMDir+"CentralModel_Best.MednAccmodel")
    _, _, _, reportBPerc = fdc.ScoreModel("CentrMednMdl", site, shareMDir, BestCentralModel,Xtest,ytest,dim,show)

    MednCentralModel = load(shareMDir+"CentralModel_Best.PercAccmodel")
    _, _, _, reportBMedn = fdc.ScoreModel("CentrBestMdl", site, shareMDir, MednCentralModel,Xtest,ytest,dim,show)



    return reportJoin, reportBPerc, reportBMedn


#function adjusted to be used in demo version
def ensembleModels(file,shareMDir):
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
