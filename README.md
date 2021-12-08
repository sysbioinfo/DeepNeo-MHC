# DeepNeo-MHC

1. Train
Arguments are allele, peptide length, gpu_num and false data type.

Support
allele: HLA-A,HLA-B, HLA-C

length: 9,10

false data type: random or natural

gpu_num: if you have more than 1 gpu, choose gpu what you want to use (default: 0)

command:
python DeepNeo_train.py HLA-A 9 random 0

2. Inference
Input DataFrame should be contained allele, length, Peptide seq
Support only 9, 10mer peptides

