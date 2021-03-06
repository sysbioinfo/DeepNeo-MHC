import pickle
import gc
import pandas as pd
from itertools import combinations, product
import platform


def call_group_list(allele):
    if allele == 'HLA-A':
        group1 = ['HLA-A-2403', 'HLA-A-2402', 'HLA-A-2413', 'HLA-A-2301', 'HLA-A-2406', 'HLA-A-2407']
        group2 = ['HLA-A-3303', 'HLA-A-3301', 'HLA-A-6801', 'HLA-A-6601', 'HLA-A-3401', 'HLA-A-6602',
                  'HLA-A-3101', 'HLA-A-7401']
        group3 = ['HLA-A-3001', 'HLA-A-0301', 'HLA-A-1101', 'HLA-A-1102', 'HLA-A-6812']
        group4 = ['HLA-A-6802', 'HLA-A-6901']
        group5 = ['HLA-A-0205', 'HLA-A-0206', 'HLA-A-0217', 'HLA-A-0216', 'HLA-A-0212', 'HLA-A-0219',
                  'HLA-A-0207', 'HLA-A-0203', 'HLA-A-0201', 'HLA-A-0211', 'HLA-A-0204', 'HLA-A-0202']
        group6 = ['HLA-A-2601', 'HLA-A-2501', 'HLA-A-2608', 'HLA-A-2603', 'HLA-A-2602']
        group7 = ['HLA-A-0103', 'HLA-A-0101', 'HLA-A-2902', 'HLA-A-3002', 'HLA-A-3601', 'HLA-A-8001']

        target_list = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']
        group_list = [group1, group2, group3, group4, group5, group6, group7]

    elif allele == 'HLA-B':
        group1 = ['HLA-B-5301', 'HLA-B-3501', 'HLA-B-3507', 'HLA-B-3508', 'HLA-B-1511']
        group2 = ['HLA-B-0704', 'HLA-B-0702', 'HLA-B-4201', 'HLA-B-3502', 'HLA-B-3503','HLA-B-3504','HLA-B-3506',]
        group3 = ['HLA-B-8101', 'HLA-B-4202',]
        group4 = ['HLA-B-5401','HLA-B-5501',]
        group5 = [ 'HLA-B-5502','HLA-B-5601']
        group6 = ['HLA-B-5101', 'HLA-B-5108', 'HLA-B-7301', 'HLA-B-3906',]
        group7 = ['HLA-B-2710', 'HLA-B-2702', 'HLA-B-2701', 'HLA-B-2704', 'HLA-B-2703', 'HLA-B-2705', 'HLA-B-2708',
                  'HLA-B-2707', 'HLA-B-2706',]
        group8 = ['HLA-B-3905', 'HLA-B-3901', 'HLA-B-3801', 'HLA-B-3802', 'HLA-B-1509', 'HLA-B-1510',]
        group9 = ['HLA-B-3924', 'HLA-B-1402', 'HLA-B-1403',]
        group10 = ['HLA-B-2709','HLA-B-3909',]
        group11 = ['HLA-B-4901', 'HLA-B-5001', 'HLA-B-4006', 'HLA-B-4101', 'HLA-B-4501',]
        group12 = ['HLA-B-1803', 'HLA-B-1801', 'HLA-B-4402', 'HLA-B-4403', 'HLA-B-4427', 'HLA-B-4428',]
        group13 = ['HLA-B-4102', 'HLA-B-4104', 'HLA-B-4103', 'HLA-B-4409', 'HLA-B-4002', 'HLA-B-4001',]
        group14 = ['HLA-B-1508','HLA-B-1501','HLA-B-1503','HLA-B-1502','HLA-B-4601',]
        group15 = ['HLA-B-5703','HLA-B-5701','HLA-B-5801', 'HLA-B-5802','HLA-B-1517',]
        group16 = ['HLA-B-5201', 'HLA-B-1302', ]
        group17 = ['HLA-B-0803', 'HLA-B-0802',]

        target_list = ['group1', 'group2', 'group3', 'group4', 'group5',
                       'group6', 'group7', 'group8', 'group9', 'group10',
                       'group11', 'group12', 'group13', 'group14', 'group15',
                       'group16', 'group17']

        group_list = [group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11,
                      group12, group13, group14, group15, group16, group17]

    else:
        group1 = ['HLA-C-0401', 'HLA-C-0501', 'HLA-C-0403', 'HLA-C-0802', ]
        group2 = ['HLA-C-1402', 'HLA-C-1402', ]
        group3 = ['HLA-C-0704', 'HLA-C-0702', 'HLA-C-0602', 'HLA-C-0701', ]
        group4 = ['HLA-C-1502', 'HLA-C-1505', ]
        group5 = ['HLA-C-1701', 'HLA-C-0801', 'HLA-C-0304', 'HLA-C-0303', 'HLA-C-1202',
                  'HLA-C-0202', 'HLA-C-1203', 'HLA-C-1601', 'HLA-C-0302', ]
        target_list = ['group1', 'group2', 'group3', 'group4', 'group5']
        group_list = [group1, group2, group3, group4, group5]

    return target_list, group_list


def load_short_hla():
    with open('short_hla_seq.pkl', 'rb') as f:
        hla = pickle.load(f)
    return hla


def load_pep_seq():
    with open('/home/jaeung/IEDB_data_filtering/MS_BA_training_set.pkl', 'rb') as f:
        df = pickle.load(f)

    del df['matrix']
    gc.collect()

    df['length'] = df['Peptide seq'].map(lambda x: len(x))
    df = df[df['length'] == 9]
    df = df[df['answer'] == 1]
    return df


def load_gradcam_result():
    if platform.system() == "Darwin":
        with open('new_short_hla_9mer_gradcam_result.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open('/home/jaeung/IEDB_data_filtering/gradcam/2021.09.21_Training_gradcam_result.pkl', 'rb') as f:
            return pickle.load(f)


def load_target_gradcam_result(allele, mode, target=0, position=0, cp='cp'):
    if cp != 'cp' and (mode == 'total' or mode == 'pattern'):
        if platform.system() == "Darwin":
            with open('/Users/jaeung/gradcam_coef_cal/data/new_short_hla_9mer_gradcam_result.pkl', 'rb') as f:
                p9_binder, _, _, _ = pickle.load(f)
        else:
            with open('/home/jaeung/IEDB_data_filtering/gradcam/2021.09.21_Training_gradcam_result.pkl', 'rb') as f:
                p9_binder = pickle.load(f)
        return p9_binder

    else:
        if platform.system() == "Darwin":
            with open(f'/Users/jaeung/gradcam_coef_cal/data/{allele}_{mode}_{target}_position_{position+1}_gradcam_result.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            if cp == 'cp':
                with open(f'/home/jaeung/Research/MHC/training_data_gradcam_result/short_{allele}_{mode}_ingroup_{position+1}_with_gradcam_result.pkl', 'rb') as f:
                    return pickle.load(f)
            else:
                with open(f'/home/jaeung/Research/MHC/{allele}_{mode}_position_{position+1}_gradcam_result_with_cp_value.pkl', 'rb') as f:
                    return pickle.load(f)


def return_group_list(group_mode, target_group_list, allele_list, allele, i):
    if group_mode == 'ingroup':
        group_list = tuple(combinations(target_group_list[i], 2))

    elif group_mode == 'outgroup':
        outgroup = tuple(set(pd.Series(allele_list)[pd.Series(allele_list).str.contains(f'{allele}')])
                         - set(target_group_list[i]))
        group_list = tuple(product(target_group_list[i], outgroup))

    return group_list


def precision_recall(y_true, y_pred):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision, recall, f1

