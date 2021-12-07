import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from benchmark_util import *
import torch
from efficientnet import *
import gc


def benchmark(model, data, result):
    BATCH_SIZE = 1024
    matrix = []
    for i in data['matrix']:
        matrix.append(i)

    matrix = np.array(matrix).astype('float32')
    dataset = torch.utils.data.TensorDataset(torch.tensor(matrix))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    predict = []
    model.to('cuda')
    model.eval()
    for i in dataloader:
        with torch.no_grad():
            predict += model((torch.Tensor(i[0]).to('cuda')))

    pred = []
    for i in predict:
        pred.append(i.cpu().numpy()[0])
    data[result] = pred


def bench_short_model(df, file_name, false_mode):
    #df = pd.read_pickle(file)
    df['length'] = df['Peptide seq'].map(lambda x: len(x))

    df_A = df[df['allele'].str.contains('HLA-A')]
    df_B = df[df['allele'].str.contains('HLA-B')]
    df_C = df[df['allele'].str.contains('HLA-C')]

    df_A_9 = df_A[df_A['length']==9]
    df_A_10 = df_A[df_A['length']==10]
    df_B_9 = df_B[df_B['length']==9]
    df_B_10 = df_B[df_B['length']==10]
    df_C_9 = df_C[df_C['length']==9]
    df_C_10 = df_C[df_C['length']==10]

    for model, data in zip((HLA_A_9, HLA_B_9, HLA_C_9, HLA_A_10, HLA_B_10, HLA_C_10), (df_A_9, df_B_9, df_C_9, df_A_10, df_B_10, df_C_10)):
        benchmark(model, data, f'DeepNeo-MHC {false_mode}') # 모델 결과 내보낼때
    df = pd.concat([df_A_9, df_B_9, df_C_9, df_A_10, df_B_10, df_C_10])

    del df['matrix']
    df.to_csv(file_name, index=False)


def load_weight(false_mode):
    if false_mode == 'random':
        # random peptide model

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-A_9_final/best_380.pth', map_location='cpu')
        HLA_A_9 = checkpoint['model']
        HLA_A_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-B_9_final/best_499.pth', map_location='cpu')
        HLA_B_9 = checkpoint['model']
        HLA_B_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-C_9_final/best_327.pth', map_location='cpu')
        HLA_C_9 = checkpoint['model']
        HLA_C_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-A_10_final/best_464.pth', map_location='cpu')
        HLA_A_10 = checkpoint['model']
        HLA_A_10.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-B_10_final/best_482.pth', map_location='cpu')
        HLA_B_10 = checkpoint['model']
        HLA_B_10.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_16_HLA-C_10_final/best_122.pth', map_location='cpu')
        HLA_C_10 = checkpoint['model']
        HLA_C_10.load_state_dict(checkpoint['state_dict'])

    elif false_mode == 'natural':
        # natural protein model
        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-A_9_final/best_485.pth',
                                map_location='cpu')
        HLA_A_9 = checkpoint['model']
        HLA_A_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-B_9_final/best_486.pth',
                                map_location='cpu')
        HLA_B_9 = checkpoint['model']
        HLA_B_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-C_9_final/best_389.pth',
                                map_location='cpu')
        HLA_C_9 = checkpoint['model']
        HLA_C_9.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-A_10_final/best_407.pth',
                                map_location='cpu')
        HLA_A_10 = checkpoint['model']
        HLA_A_10.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-B_10_final/best_482.pth',
                                map_location='cpu')
        HLA_B_10 = checkpoint['model']
        HLA_B_10.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load('saved_model/DeepNeo_Sep_18_natural_protein_HLA-C_10_final/best_39.pth',
                                map_location='cpu')
        HLA_C_10 = checkpoint['model']
        HLA_C_10.load_state_dict(checkpoint['state_dict'])

    return HLA_A_9, HLA_B_9, HLA_C_9, HLA_A_10, HLA_B_10, HLA_C_10


if __name__ == "__main__":
    # DataFrame should be contained allele, length, Peptide seq
    # support only 9, 10mer peptides
    data = pd.read_pickle('2021.09.16_IEDB_Testset.pkl')
    false_mode = 'random'  # choose random or natural model
    HLA_A_9, HLA_B_9, HLA_C_9, HLA_A_10, HLA_B_10, HLA_C_10 = load_weight(false_mode)

    # make input 2d matrix
    df = use_multicore(data, multiprocessing.cpu_count())  # type nums of cpu cores how many you want
    # inference, you can change result file name
    bench_short_model(data, file_name='result.pkl', false_mode=false_mode)



