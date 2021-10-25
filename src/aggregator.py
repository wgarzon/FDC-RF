import os
from joblib import load
import sys
from copy import deepcopy

# #get information about share model from another sites, excluding local site
# def get_sharedModels(path, locSite, ext):
#     #get complete model but exclude local model only from others
#     modelFiles = [path + i for i in os.listdir(path) if i.endswith(ext) and i.count(locSite)==0]
#     models = []
#     if len(modelFiles) == 0: sys.exit("!!!There aren't files '." + ext + "'  to upload!!!")
#     #load each model to merge with local models
#     for mod in modelFiles: models.append(load(mod))
#     globModel = reduce(combine_mods, models)
#     return globModel

#get information about models and features shared by each tree
def get_models(path, ext):
    modelFiles = [path + i for i in os.listdir(path) if i.endswith(ext)]
    bestMdl, badMdl, globBadMdls = [], [], []
    if len(modelFiles) == 0: sys.exit("!!!There aren't files '." + ext + "'  to upload!!!")
    #load each model to merge with local models
    for mod in modelFiles:
        #if mod.count("Best") > 0: bestMdl.append(load(mod))
        if mod.count("Best") > 0:
            bestMdl.append(load(mod))
        else:
            badMdl.append(load(mod))

    #globBestMdls = reduce(combine_mods, bestMdl)
    globBestMdls = combine_mods2(bestMdl)
    #globBadMdls = reduce(combine_mods, badMdl)
    return globBestMdls, globBadMdls

#get information about share model from another sites, excluding local site
def get_siteModels(path, ext):
    modelFiles, models = [path + i for i in os.listdir(path) if i.endswith(ext)], []
    if len(modelFiles) == 0: sys.exit("!!!There aren't files '." + ext + "'  to upload!!!")
    #load each model to merge with local models
    for mod in modelFiles:  models.append(load(mod))
    #globModel = reduce(combine_mods, models)
    globModel = combine_mods2(models)
    return globModel

#Join the models of many sites from site1 (site ensembler)
def mergeModelsJoin(pathShare, ext):
    modShared = get_siteModels(pathShare, ext)
    return modShared

def mergeModelsAcc(pathShare, ext):
    bestMdl, badMdl = get_models(pathShare, ext)
    return bestMdl

def combine_mods2(RFModels):
    nModel = deepcopy(RFModels[0])
    for val in range(1,len(RFModels)):
        nModel.estimators_ += RFModels[val].estimators_
        nModel.n_estimators = len(nModel.estimators_)

    return nModel

def combine_mods(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

#get information about share model from another sites, excluding local site
def get_centModel(path):
    central  = load(path)
    return central
