import numpy as np
from Evaluation import evaluation, Net_evaluation
from Global_Vars import Global_vars
from Model_Transunet import Model_TransUnet


def Objective_Seg(Soln):
    Feat = Global_vars.Feat
    Target = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            pred = Model_TransUnet(Feat, sol)
            Eval = Net_evaluation(pred, Target)
            Fitn[i] = 1 / (Eval[3])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        pred = Model_TransUnet(Feat, sol)
        Eval = Net_evaluation(pred, Target)
        Fitn = 1 / (Eval[3])
        return Fitn


def Obj_fun(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = (1 / (Eval[4] + Eval[13] + Eval[10])) + Eval[9]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = (1 / (Eval[4] + Eval[13] + Eval[10])) + Eval[9]
        return Fitn



