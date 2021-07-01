import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
def eval(csvfile):
    result_pd=pd.read_csv(csvfile,header=None)
    non_covid_list=result_pd[0].str.contains('_non-covid')
    result_pd.loc[non_covid_list,0 ] = 0
    result_pd.loc[~non_covid_list,0] = 1
    gt=result_pd[0].values.astype(np.uint8)
    pred = result_pd[1].values.astype(np.uint8)
    return {'F1':f1_score(gt,pred),
            'accuracy':accuracy_score(gt,pred),
            'precision':precision_score(gt,pred),
            'recall':recall_score(gt,pred)}

if __name__ == '__main__':
    exp_dir = 'experimentsThan'
    result_dir = os.path.join(exp_dir, 'result')
    test_result_m = os.path.join(result_dir, 'test_result_m.txt')
    result=eval(test_result_m)
    print(result)