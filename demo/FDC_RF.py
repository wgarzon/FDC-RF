from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import utils
import numpy as geek
from datetime import datetime
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

# Create dictionary of models with according object
# "Balanced RF": Pipeline([('clf', BalancedRandomForestClassifier())]),
# "Balanced RF": Pipeline([('clf', BalancedRandomForestClassifier())]),
Models = {"Random Forest": Pipeline([('clf', RandomForestClassifier())]),
          "Bagging": Pipeline([('clf', BaggingClassifier())]),
          "Balanced RF": Pipeline([('clf', RandomForestClassifier())]),
          "Bootstrap Class Weighting RF": Pipeline([('clf', RandomForestClassifier())]),
          "Balanced Bagging": Pipeline([('clf', BalancedBaggingClassifier())])}


# Create dictionary with parameters of models
#INIT ModelParam = {"Random Forest": {"n_estimators": 100, "max_features": "log2", "oob_score": True, "n_jobs": -1},
#1.8GB ModelParam = {"Random Forest": {"n_estimators": 100, "min_samples_split": 12, "min_samples_leaf": 4, "max_features": "auto", "oob_score": True, "n_jobs": -1},
#ModelParam = {"Random Forest": {"n_estimators": 100, "min_samples_split": 12, "min_samples_leaf": 4, "max_features": "auto", 'max_depth': 110, "oob_score": True, "n_jobs": -1},
ModelParam = {"Random Forest": {"n_estimators": 1000, "min_samples_split": 12, "min_samples_leaf": 4, "max_features": 0.25, "oob_score": True, "n_jobs": -1},
              "Bagging": {"base_estimator": [DecisionTreeClassifier(max_features="auto", splitter="random", max_leaf_nodes=16)],"n_estimators": 100, "bootstrap": True, "oob_score": True, "n_jobs": -1},
              "Balanced RF": {"n_estimators": 100, "class_weight": "balanced", "max_features": "sqrt", "n_jobs": -1},
              "Bootstrap Class Weighting RF": {"n_estimators": 100, "class_weight": "balanced_subsample", "n_jobs": -1},
              "Balanced Bagging": {"base_estimator": [DecisionTreeClassifier(max_features="auto", splitter="random", max_leaf_nodes=16)], "n_estimators": 100, "n_jobs": -1}}

ModelParams = {"Random Forest": {"n_estimators": [50, 100],"criterion": ["gini", "entropy"],"max_leaf_nodes": [20, 300], "oob_score": [True], "n_jobs": [-1]},
               "Bagging": {"base_estimator": [DecisionTreeClassifier(max_features="auto", splitter="random", max_leaf_nodes=16)],"n_estimators": [100, 500, 600, 800],"bootstrap": [True],"oob_score": [True],"n_jobs": [-1]},
               "Balanced RF": {"n_estimators": [200], "class_weight": ["balanced"],"max_features": ["auto", "sqrt", "log2"],"max_depth": [3, 4, 5, 6, 7, 8],"min_samples_split": [0.005, 0.01, 0.05, 0.10],"min_samples_leaf": [0.005, 0.01, 0.05, 0.10],"criterion": ["gini", "entropy"],"n_jobs": [-1]},
               "Bootstrap Class Weighting RF": {"n_estimators": [200],"class_weight": ["balanced_subsample"],"max_features": ["auto", "sqrt", "log2"],"max_depth": [3, 4, 5, 6, 7, 8],"min_samples_split": [0.005, 0.01, 0.05, 0.10],"min_samples_leaf": [0.005, 0.01, 0.05, 0.10],"criterion": ["gini", "entropy"],"n_jobs": [-1]},
               "Balanced Bagging": {"base_estimator": [DecisionTreeClassifier(max_features="auto", splitter="random", max_leaf_nodes=16)],"n_estimators": [100, 500, 600, 800],"max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],"n_jobs": [-1]}}


class FDC_RF():
    def __init__(self, mdlName):
        self.pipelines = Models[mdlName]
        self.modelName = mdlName
        self.model = None
        self.nFeats = 0

    def getModel(self):
        return self.pipelines[0]

    def getMdlName(self):
        return self.modelName

    def getNoFeats(self):
        return self.nFeats


def buildModel(mdlName,DTs, Xtrain, ytrain):
    fdc_rf = FDC_RF(mdlName)

    model = fdc_rf.getModel()
    vals = ModelParam[mdlName]
    vals['n_estimators'] = DTs
    model.set_params(**vals)

    model.fit(Xtrain, ytrain)
    fdc_rf.model = model
    fdc_rf.nFeats = pd.DataFrame(Xtrain).shape[1]


    return fdc_rf

# receive a model to obtain the relevant data for each DT
# infoM1 has the format: tree, usedFeats
# infoM2 has the format: binFeatVector
def getInfoDTs(model, feats):
    # create a list with zero values
    infoM1, infoM2, k, maxLen = {}, {}, 1, 0
    vecFeats = [0] * feats

    # estimators_: List of DecisionTreeClassifier
    for ind_tree in model.estimators_:
        # print("positiveFeats",k,ind_tree.tree_.feature)
        positiveFeats = list(filter(lambda x: x > 0, ind_tree.tree_.feature))
        # create a binary vector with features used during tree construction
        x, x[positiveFeats] = np.array(vecFeats), 1
        # the dict has the format: tree, usedFeats, binFeatVector
        #print("ID Features used:", k, positiveFeats)
        if maxLen < max(positiveFeats): maxLen = max(positiveFeats)
        infoM1[k], infoM2[k], k = [ind_tree.tree_, positiveFeats], x, k + 1

    return infoM1, infoM2, maxLen


# It is better to have lower OOOerror rate per tree: 0:Good, 1:Bad
def ParOOBErrorTree(model, X, y):
    typ = "MSE"
    n_samples = X.shape[0]

    OOB_Err, OOB_Acc = {}, {}
    aggAcc = 0

    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    print("Number of cores:", cores)
    model.n_jobs = -1

    # Here at each iteration we obtain out of bag samples for every tree.
    results = pool.starmap(computeError, [(tree, n_samples, X, y, typ) for tree in model.estimators_])
    pool.close()

    lenRF = len(results)

    for idx in range(lenRF):
        OOB_Err[idx + 1], OOB_Acc[idx + 1] = results[idx][0], results[idx][1]
        aggAcc += results[idx][1]

    aggAcc = round(aggAcc / lenRF, 3)
    return OOB_Err, OOB_Acc, aggAcc


def computeError(tree, n_samples,X, y, typ="MSE"):
    unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples,n_samples)
    #get info for dataset and classes for OOB indices
    OOB_DSet = X.iloc[unsampled_indices,:]
    OOB_Y = y.values[unsampled_indices]
    #make the prediction for bag sample indices Predicting class probabilities
    predic = tree.predict(OOB_DSet)
    #probs = tree.predict_proba(OOB_DSet)
    if typ == "MSE":
        error = sum(abs(OOB_Y-predic))/len(predic)
        #error = mean_squared_error(OOB_Y, predic)
    else:
        error = r2_score(OOB_Y, predic)

    #[OOB_Err,OOB_Acc]
    return [round(error,2), round(1-error,2)]


#Get a % of the best DTs
# Errors {1: 0.68, 2: 0.76, 3: 0.16}
def getBestByPerc(Accu, perc):
    best, bad = [],[]
    vals = list(Accu.values())
    vals.sort(reverse=True)
    vals = vals[0: int(len(vals)*perc)]

    for DT in Accu:
        if Accu[DT] in vals:
            best.append(DT)
        else:
            bad.append(DT)

    DTClusters = {key:value for key, value in zip([1,0], [best,bad])}
    return DTClusters, vals[-1]

#Get best models take the median
def getBestByMedian(Accu):
    best, bad = [],[]
    vals = list(Accu.values())
    median = statistics.median(vals)
    vals.sort(reverse=True)
    vals = [a for a in vals if a >= median]

    for DT in Accu:
        if Accu[DT] in vals:
            best.append(DT)
        else:
            bad.append(DT)

    DTClusters = {key:value for key, value in zip([1,0], [best,bad])}
    return DTClusters, median

#Get best DT according to a threshold value
def getBestByThres(Accu, Thresh):
    Up, Dw = [],[]

    for DT in Accu:
        if Accu[DT]>=Thresh:
            Up.append(DT)
        else:
            Dw.append(DT)

    DTClusters = {key:value for key, value in zip([1,0], [Up,Dw])}
    return DTClusters


# It is better to have lower OOOerror rate per tree. 0:Good, 1:Bad
def getOOBErrorTree(model, X, y):
    n_samples = X.shape[0]
    n_outputs_ = len(y)
    OOB_Err, OOB_Acc, IdT = {}, {}, 1
    model.n_jobs = -1
    # Here at each iteration we obtain out of bag samples for every tree.
    for tree in model.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples)
        # get info for dataset and classes for OOB indices
        OOB_DSet = X.iloc[unsampled_indices, :]
        OOB_Y = y.values[unsampled_indices]
        # make the prediction for bag sample indices Predicting class probabilities
        predic = tree.predict(OOB_DSet)
        # probs = tree.predict_proba(OOB_DSet)
        error = sum(abs(OOB_Y - predic)) / len(predic)
        OOB_Err[IdT], OOB_Acc[IdT] = round(error, 2), round(1 - error, 2)

        IdT += 1

    return OOB_Err, OOB_Acc

# Stats about the trees in random forest. List of DecisionTreeClassifier
def ShowInfoDTs(model,site):
    n_nodes, max_depths, k = [], [], 1
    print("*** Information of Decision Trees ***")
    # estimators_: List of DecisionTreeClassifier
    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)
        utils.saveTree(k,ind_tree,site)
        k+=1

    print(f'Average number of nodes {int(np.mean(n_nodes))}')
    print(f'Average maximum depth {int(np.mean(max_depths))}')


# Create models according to No. Clusters
# {'Best (1)': [4, 7, 8, 10], 'Bad (0)': [1, 2, 3, 5, 6, 9]}
def splitMdlsByAcc(clusters, OriModel, data,modelName):
    ClusModels = {}

    for id_CL in clusters:
        valClusters = list(clusters[id_CL])

        nModel = RandomForestClassifier()
        nvals = ModelParam[modelName].copy()
        nvals['n_estimators'] = len(valClusters)
        nModel.set_params(**nvals)
        nModel.fit(data[0], data[1])

        k=0
        for id_DT in valClusters:
            nModel.estimators_[k].tree_ = OriModel.estimators_[id_DT-1].tree_
            k+=1

        ClusModels[id_CL] = nModel
        print("No. nModel:", id_CL,len(nModel.estimators_))

    return ClusModels


def accScore(ytest, y_pred):
    ytest = ytest.tolist()
    simil = len([i for i, j in zip(ytest, y_pred) if i == j])
    perc = simil/len(ytest)
    return perc

#Original version
#def ScoreModel(site, idsite, path, model, Xtest, ytest,dim):
#    try:
        #y_pred = model.predict(Xtest)
        # score2 = accScore(ytest, y_pred)
#        y_pred = PPredict(model, Xtest)
#        score = accuracy_score(ytest, y_pred)
#        print("*** Score Model " + site + " ***")
#        estimators = len(model.estimators_)
#        oob_score = model.oob_score_
#        text = '#DTs:{0},Model Score:{1},OOB Score:{2}'.format(estimators, score, oob_score)
#        print(text)
#        ff = open(path+'outcomes/Results_'+idsite+'.txt', 'a')
#        ff.write(site + ","+ text + "," + dim +","+ str(datetime.now().strftime("%d%m%Y_%H%M")) + "\n")

#        return score, estimators, oob_score
#    except ValueError:
#        print('The number of features of the model must match the input!!!')

#new version
def ScoreModel(site, idsite, path, model, Xtest, ytest,dim,show):
    try:
        #y_pred = model.predict(Xtest)
        # score2 = accScore(ytest, y_pred)
        y_pred = PPredict(model, Xtest)

        score = "accSc:" + str(accuracy_score(ytest, y_pred))
        score = score  + ",a1:" + str(f1_score(ytest, y_pred, average='macro'))
        score = score  + ",b1:" + str(f1_score(ytest, y_pred, average='micro'))
        score = score  + ",c1:" + str(f1_score(ytest, y_pred, average='weighted'))

        score = score  + ",a2:" + str(precision_score(ytest, y_pred, average='macro'))
        score = score  + ",b2:" + str(precision_score(ytest, y_pred, average='micro'))
        score = score  + ",c2:" + str(precision_score(ytest, y_pred, average='weighted'))

        score = score  + ",a3:" + str(recall_score(ytest, y_pred, average='macro'))
        score = score  + ",b3:" + str(recall_score(ytest, y_pred, average='micro'))
        score = score  + ",c3:" + str(recall_score(ytest, y_pred, average='weighted'))

        estimators = len(model.estimators_)
        oob_score = model.oob_score_
        text = '#DTs:{0},Model Score:{1},OOB Score:{2}'.format(estimators, score, oob_score)
        if show:
            print("*** Score Model " + site + " ***")
            print(text)
            print(classification_report(ytest,y_pred))

        ff = open(path+'outcomes/Results_'+idsite+'.txt', 'a')
        ff.write(site + ","+ text + "," + dim +","+ str(datetime.now().strftime("%d%m%Y_%H%M")) + "\n")

        return score, estimators, oob_score, classification_report(ytest,y_pred,output_dict=True)
    except ValueError:
        print('The number of features of the model must match the input!!!')


# falta tener en cuenta la distrib de clases
def PPredict(model, Xtest, classes=5):
    data = Xtest.values
    zerosMat = geek.zeros([len(data), classes])
    XtestFloat32 = data.astype(np.float32)

    print("No. Estimators",len(model.estimators_))
    for eleme in model.estimators_:
        y_pred = eleme.tree_.predict(XtestFloat32)
        zerosMat += y_pred

    predict = []
    for row in zerosMat:
        a = row.tolist()
        predict.append(a.index(max(a)))

    return predict
