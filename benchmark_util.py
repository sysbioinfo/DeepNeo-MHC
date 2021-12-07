import numpy as np
import pandas as pd
import multiprocessing
import parmap

    
def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores) # 코어 수 만큼 데이터 프레임 쪼개기
    data = parmap.map(func, df_split, pm_pbar=True, pm_processes=num_cores)
    try:
        data = pd.concat(data, ignore_index=True)
        return data
    except:
        return data


def make_df(df):
    df['matrix'] = df.apply(make_matrix_short_hla, axis = 1)
    return df


def use_multicore(df, num_cores = multiprocessing.cpu_count()):
    df = parallelize_dataframe(df, make_df, num_cores)
    return df


def load_short_hlaseq():
    hla_a_prot = pd.read_csv('HLA_A_prot.txt', sep='\t', header = None)
    hla_b_prot = pd.read_csv('HLA_B_prot.txt', sep='\t', header = None)
    hla_c_prot = pd.read_csv('HLA_C_prot.txt', sep='\t', header = None)
    
    hla_a_prot[1] = hla_a_prot[1].map(lambda x: x[24:-65])
    hla_c_prot[1] = hla_c_prot[1].map(lambda x: x[4:-66])
    hla_b_prot[1] = hla_b_prot[1].map(lambda x: x[12:-62])
    
    hla_prot = pd.concat([hla_a_prot, hla_b_prot, hla_c_prot], axis = 0)
    hla = {}
    for line in hla_prot.to_numpy():
        hla[line[0]] = line[1]
        
    return hla


def make_matrix_short_hla(df):
    # 해시값 불러오기
    hash_data = pd.read_csv('Calpha.txt', sep='\t')
    hash_data.set_index(hash_data['Unnamed: 0'], inplace=True)
    # 불필요한 칼럼 정리
    del hash_data['Unnamed: 0']
    hash_data = np.exp(-1*hash_data)
    hash_data_list = []
    hla = load_short_hlaseq()
    
    for amino in hla[df['allele']]:
        if amino in hash_data.columns:
            for target in df['Peptide seq']:
                if target == '*' or target == 'X' or target == 'U':
                    hash_data_list.append(0)
                else:
                    hash_data_list.append(hash_data[amino][target])
        else:
            for _ in df['Peptide seq']:
                hash_data_list.append(0)
    
    if 'HLA-A' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,276,9).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,252,9).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==9:
        matrix = np.array(hash_data_list).reshape(1,268,9).astype('float32')
    elif 'HLA-A' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,276,10).astype('float32')
    elif 'HLA-B' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,252,10).astype('float32')
    elif 'HLA-C' in df['allele'] and len(df['Peptide seq'])==10:
        matrix = np.array(hash_data_list).reshape(1,268,10).astype('float32')

    return matrix

