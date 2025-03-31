import ast
import numpy as np
import pandas as pd
from sklearn import metrics
print(pd.__version__)
def evaluation():
    exp_subspace_list = pd.read_csv('datasets/ground_truth/WDBC_gt_hbos.csv')
    print("exp_subspace_list", exp_subspace_list)

    gt_subspace = pd.read_csv('datasets/results/WDBC_pre.csv')
    print("gt_subspace", gt_subspace)

    #ano_idx = [3,32,70,72,93,179,192,194,213]
    #ano_idx = [18,38,41,45,73,79,94,151,161,167,170,199,224,261,266,409,459,517,573,576,600,633,647,659,690,703,704,724,813,830,832,833,872,876,899,927,937,1124,1176,1189,1233,1235,1238,1239,1261,1263,1276,1293,1299,1307,1363,1369,1374,1423,1461,1467,1469,1478,1480,1482,1484,1505,1521]

    #ano_idx = len(exp_subspace_list)
    #ano_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    #ano_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    #           41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,4329,4331,4332,
    #           4340,4342,4344,4347,4349,4358,4362,4363,4364,4369,4372,4374,4375,4376,4381,4383,4385,4386,4387,4390,4394,4397,4401,
    #           4403,4404,4406,4408,4409,4410,4412,4413,4415,4419,4423,4427,4429,4430,4431,4432,4434,4435,4436,4437,4438,4439,4440,
    #           4441,4442,4443,4444,4445,4452,4456,4459,4460,4461,4462,4463,4469,4471,4474,4475,4478,4485,4486,4487,4490,4498,4499,
    #           4500,4502,4508,4511,4512,4513,4514,4515,4517,4518,4519,4520,4521,4522,4523,4525,4527,4528,4532,4533,4535,4542,4543,
    #           4544,4546,4547,4548,4549,4550,4556,4557,4560,4561,4562,4563,4568,4569,4570,4572,4573,4575,4576,4580,4582,4583,4589,
    #           4590,4591,4592,4593,4594,4596,4598,4599,4600,4603,4608,4611,4612,4613,4614,4618,4619,4621,4622,4626,4630,4634,4635,
    #           4636,4637,4638,4641,4642,4643,4644,4645,4647,4648,4650,4651,4656,4658,4661,4662,4667,4668,4669,4670,4671,4677,4678,
    #           4679,4680,4681,4682,4683,4686,4687,4690,4694,4699,4700,4702,4706,4707,4708,4709,4710,4727,4734]
    ano_idx = [0,1,2,3,4,5,6,7,8,9]
    #ano_idx = [43,44,45,103,132,147]
    #ano_idx = [18,51,54,55,58,60,61,66,68,71,74,76,79]
    precision_list = np.zeros(len(ano_idx))
    jaccard_list = np.zeros(len(ano_idx))
    recall_list = np.zeros(len(ano_idx))

    for ii, ano in enumerate(ano_idx):
        exp_subspace_list_str = exp_subspace_list.loc[exp_subspace_list["ano_idx"] == ano, "exp_subspace"].values[0]
        exp_subspace = ast.literal_eval(exp_subspace_list_str)
        print("exp_subspace", exp_subspace)

        gt_subspace_str = gt_subspace.loc[gt_subspace["ano_idx"] == ano, "exp_subspace"].values[0]
        print("gt_subspace_str", gt_subspace_str)
        g_subspace = ast.literal_eval(gt_subspace_str)

        overlap = list(set(g_subspace).intersection(set(exp_subspace)))
        union = list(set(g_subspace).union(set(exp_subspace)))

        precision_list[ii] = len(overlap) / len(exp_subspace)
        jaccard_list[ii] = len(overlap) / len(union)
        recall_list[ii] = len(overlap) / len(g_subspace)
    print("F1", (2*precision_list.mean() * recall_list.mean())/(precision_list.mean() + recall_list.mean()))

    return precision_list.mean(), recall_list.mean(), jaccard_list.mean()
def evaluation_auc(in_shape):
    
    exp_subspace_list = pd.read_csv('datasets/ground_truth/glass_iforest.csv')

    gt_subspace = pd.read_csv('datasets/results/glass_pre.csv')
    ano_idx = [43,44,45,103,132,147]
    auroc_list = np.zeros(len(ano_idx))
    aupr_list = np.zeros(len(ano_idx))
    for ii, ano in enumerate(ano_idx):
        score = feature_weight[ii]

        # ground_truth metrics
        gt_subspace_str = gt_subspace.loc[gt_subspace["ano_idx"] == ano]["exp_subspace"].values[0]
        gt_subspace = ast.literal_eval(gt_subspace_str)
        gt = np.zeros(in_shape, dtype=int)
        gt[gt_subspace] = 1

        if len(gt_subspace) == in_shape:
            auroc_list[ii] = 1
            aupr_list[ii] = 1
        else:
            precision, recall, _ = metrics.precision_recall_curve(gt, score)
            aupr_list[ii] = metrics.auc(recall, precision)
            auroc_list[ii] = metrics.roc_auc_score(gt, score)

    return aupr_list.mean(), auroc_list.mean()



if __name__ == '__main__':
    print('Start test...')
    evaluation()
    print('test completed')