# -*- coding: utf-8 -*-

import os
import time
import math
import torch
import scipy
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def read_traitsF(traitsF):  
    ### read the .csv file with traits (subjects_row x traits_column)
    traits = pd.read_csv(traitsF,
                         index_col=0) 
    subIDs = list(traits.index)
    traitIDs = traits.columns
    traits = np.array(traits,
                      dtype=np.int16)  
    return (subIDs, traitIDs, traits)  


def read_variantsF1(variantsF):  
    ## read the large genomic file (row_SNPs x column_subjects) using pandas
    df = pd.read_csv(variantsF,
                     index_col=0)  
    varIDs = list(df.index)
    subIDs = list(int(x) for x in df.columns)
    variants = np.array(df, dtype=np.int8)  
    A0 = np.ones(len(subIDs), dtype=np.int8)
    variants = np.row_stack((A0, variants))
    varIDs.insert(0, 'A0')
    df = pd.DataFrame(variants, index=varIDs, columns=subIDs, dtype=np.int8)
    return (subIDs, varIDs, variants, df)  


def Finding_RootNode(traits, variants):
    #Get max_value, max_index to find RootNode
    variants_sum = variants.sum(axis=1).reshape(-1, 1)
    traits_sum = traits.sum(axis=0).reshape(1, -1)
    V1D1 = variants @ traits
    V1D0 = torch.ones_like(V1D1) * variants_sum - V1D1
    V0D1 = torch.ones_like(V1D1) * traits_sum - V1D1
    V0D0 = torch.ones_like(V1D0) * variants.shape[1] - torch.ones_like(V1D0) * traits_sum - V1D0
    cp_D1V1 = (1 + V1D1) / (2 + V1D1 + V1D0) * 1.0
    cp_D1V0 = (1 + V0D1) / (2 + V0D1 + V0D0) * 1.0
    # RR is risk ratio
    RR = cp_D1V1 / cp_D1V0

    # Calculate the log Marginal Likelihood for this particular SNP
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))

    if variants.ndim == 1:
        lgM = lgM.reshape(1, lgM.shape[0])

    # get the maximum value and index
    max_value = torch.max(lgM, dim=0).values
    max_index = torch.max(lgM, dim=0).indices
    return RR, lgM, max_value, max_index


def root_node_function(sga_pheno, sga_var, var, variants, q=0.75, device='cpu'):
    # Get RootNode's 2by2 table, odds ratio and lgM statistic
    sga_var_sum = sga_var.sum()
    sga_pheno_sum = sga_pheno.sum()
    v1d1 = sga_var @ sga_pheno
    v1d0 = torch.ones_like(v1d1) * sga_var_sum - v1d1
    v0d1 = sga_pheno_sum - v1d1
    v0d0 = torch.ones_like(v1d0) * len(sga_var) - sga_pheno_sum - v1d0

    trace = 0

    # when j=0 (V=0)
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + v0d1 + v0d0)
    lgM += torch.lgamma(1.0 + v0d0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + v0d1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + v1d1 + v1d0)
    lgM += torch.lgamma(1.0 + v1d0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + v1d1) - torch.lgamma(torch.tensor(1.0))

    # Calculate the log Marginal Likelihood and odds ratio
    length = trace + 1
    structure_prior = length * torch.log(torch.tensor(q, device=device))
    lgM += structure_prior
    lgM = lgM.cpu().numpy().item()
    v1d1 = v1d1.cpu().numpy().item()
    v1d0 = v1d0.cpu().numpy().item()
    v0d1 = v0d1.cpu().numpy().item()
    v0d0 = v0d0.cpu().numpy().item()
    try:
        OR_r = (v1d1 / v1d0) / (v0d1 / v0d0)
    except:
        OR_r = torch.nan
    OR_details = [var, v1d1, v1d0, v0d1, v0d0, OR_r]

    return lgM, OR_details


def DriverSearch(weights, traits, variants, length, q, device):
    # Get the log Marginal Likelihood for all variants

    BP_V = weights.T * traits
    BP_V_sum = BP_V.sum(axis=0).reshape(1, -1)
    BP_V_xor = (torch.ones(traits.shape, device=device, dtype=torch.float64) - traits) * weights.T
    BP_V_xor_sum = BP_V_xor.sum(axis=0).reshape(1, -1)

    V1D1 = variants @ BP_V
    V1D0 = variants @ BP_V_xor
    V0D1 = torch.ones_like(V1D1) * BP_V_sum - V1D1
    V0D0 = torch.ones_like(V1D0) * BP_V_xor_sum - V1D0

    # when j=0 (V=0)
    lgM = torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V0D1 + V0D0)
    lgM += torch.lgamma(1.0 + V0D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V0D1) - torch.lgamma(torch.tensor(1.0))

    # when j=1 (V=1)
    lgM += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(2.0 + V1D1 + V1D0)
    lgM += torch.lgamma(1.0 + V1D0) - torch.lgamma(torch.tensor(1.0))
    lgM += torch.lgamma(1.0 + V1D1) - torch.lgamma(torch.tensor(1.0))

    if variants.ndim == 1:
        lgM = lgM.reshape(1, lgM.shape[0])
    structure_prior = length * torch.log(torch.tensor(q, device=device))
    lgM = lgM + structure_prior

    return lgM, (V1D1, V1D0, V0D1, V0D0)


def get_or_details(snp, x, index):
	# Get the odds ratio for all variants
    V1D1 = x[0][index].cpu().numpy()[0]
    V1D0 = x[1][index].cpu().numpy()[0]
    V0D1 = x[2][index].cpu().numpy()[0]
    V0D0 = x[3][index].cpu().numpy()[0]
    try:
        OR_r = (V1D1 / V1D0) / (V0D1 / V0D0)
    except:
        OR_r = np.nan
    OR_details = [snp[0], V1D1, V1D0, V0D1, V0D0, OR_r]
    return OR_details


def dt(varID, Neg_infinity, trace, index, prev_weights, root_lgM, OR_details=[], collect=[], pointer=[], q=0.75,
       device="cpu"):
    # IBI-DT function -(using decsision tree)

    warnings.filterwarnings('ignore')
    Best_score = Neg_infinity
    i = varIDs.index(varID)
    left_or_right = index

    if index == 2:
        # index = 2 is for root node
        # Get best variable, corresponding score and odds ratio
        Best_variable = varID
        OR_details = OR_details
        Best_score = root_lgM

        # Calculate and find maximum LgM for Child Left Node (variant==0)
        variants_selected = variants[i, :].reshape(1, -1)
        weights00 = torch.ones(variants_selected.shape, dtype=torch.float64, device=device) - variants_selected
        length = trace + 2
        lgMv00 = DriverSearch(weights00, traits, variants, length, q, device)[0]
        new_score_0 = torch.max(lgMv00, dim=0).values.cpu().numpy()[0]  # Score for Child Left Node

        # Calculate and find maximum LgM for Child Right Node (variant==1)
        weights11 = variants_selected
        length = trace + 2
        lgMv11 = DriverSearch(weights11, traits, variants, length, q, device)[0]
        new_score_1 = torch.max(lgMv11, dim=0).values.cpu().numpy()[0]  # Score for Child Right Node

        # New Score while split into two Branches
        New_score = new_score_0 + new_score_1

        # Condition to stop or continue
        if New_score > Best_score:
            trace = trace + 1
            collect = collect
            collect.append(trace)
            _list = [Best_score, Best_variable, trace, left_or_right, OR_details,
                     New_score]
            pointer.append(_list)
            ptr0 = dt(Best_variable, new_score_0, trace, 0, weights00, root_lgM, OR_details, collect, pointer,
                      q, device)
            ptr1 = dt(Best_variable, new_score_1, trace, 1, weights11, root_lgM, OR_details, collect, pointer,
                      q, device)

        else:
            _list = [Best_score, Best_variable, trace, left_or_right, OR_details, New_score]
            pointer.append(_list)
            return (pointer)
    else:
        # index = 0 is for left node
        if index == 0 and trace >= 1:
            Best_score = Neg_infinity
            weights_0 = prev_weights
            length = trace + 1

            # Find best variable
            lgMv0, x = DriverSearch(weights_0, traits, variants, length, q, device)
            lgMv0_and_1 = lgMv0
            lgMv0_sGD = torch.max(lgMv0_and_1, dim=0).values.cpu().numpy()[0]
            sGD_index = torch.max(lgMv0_and_1, dim=0).indices
            sGD = []
            for item in sGD_index:
                sGD.append(varIDs[item])
            sGD = np.array(sGD)
            sorted=lgMv0_and_1.sort(descending=True,dim=0)

            previous_weights = prev_weights
            j = sGD_index[0]

            # Calculate and find maximum LgM for Child Left Node (variant==0)
            current_node_variant_selected = variants[j].reshape(1, -1)
            current_node_weights00 = torch.ones(current_node_variant_selected.shape, dtype=torch.float64,
                                                device=device) - current_node_variant_selected
            weights00 = previous_weights * current_node_weights00
            length = trace + 2
            lgMv00 = DriverSearch(weights00, traits, variants, length, q, device)[0]
            new_score_0 = torch.max(lgMv00, dim=0).values.cpu().numpy()[0]

            # Calculate and find maximum LgM for Child Right Node (variant==1)
            current_node_weights11 = current_node_variant_selected
            weights11 = previous_weights * current_node_weights11
            length = trace + 2
            lgMv11 = DriverSearch(weights11, traits, variants, length, q, device)[0]
            new_score_1 = torch.max(lgMv11, dim=0).values.cpu().numpy()[0]  # Score for Child Right Node

            # New Score while split into two Branches
            New_score = new_score_0 + new_score_1

            # Condition to stop or continue
            if New_score > Best_score:
                trace = trace + 1
                collect = collect
                collect.append(trace)
                Best_variable = sGD[0]
                Best_variable_index = sGD_index[0]
                snp = [Best_variable]

                # Get corresponding odds ratio and 2 by 2 table
                OR_details = get_or_details(snp, x, j)

                _list = [Best_score, Best_variable, trace, left_or_right, OR_details,
                         New_score]
                pointer.append(_list)
                print("index:{},best_variable:{},new_score:{},best_score:{},new_score_0:{},new_score_1:{},length:{}".
                      format(left_or_right, Best_variable, New_score, Best_score, new_score_0, new_score_1,length))
                ptr0 = dt(Best_variable, new_score_0, trace, 0, weights00, root_lgM, OR_details, collect, pointer,
                          q, device)
                ptr1 = dt(Best_variable, new_score_1, trace, 1, weights11, root_lgM, OR_details, collect, pointer,
                          q, device)

            else:
                bv = 0
                try:
                    snp = [varID]
                    # Get corresponding odds ratios and 2 by 2 table
                    _details = get_or_details(snp, x, i)
                except:
                    _details = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                _list = [Best_score, bv, trace, left_or_right, _details, New_score]
                print("index:{},best_variable:{},new_score:{},best_score:{},new_score_0:{},new_score_1:{},length:{}".
                      format(left_or_right, bv, New_score, Best_score, new_score_0, new_score_1,length))
                pointer.append(_list)

        if index == 1 and trace >= 1:
            # index = 1 is for right node
            Best_score = Neg_infinity
            weights_1 = prev_weights
            length = trace + 1

            # Find best variable
            lgMv1, x = DriverSearch(weights_1, traits, variants, length, q, device)
            lgMv0_and_1 = lgMv1
            lgMv0_sGD = torch.max(lgMv0_and_1, dim=0).values.cpu().numpy()[0]

            sGD_index = torch.max(lgMv0_and_1, dim=0).indices
            sorted=lgMv0_and_1.sort(descending=True,dim=0)
            sGD = []
            for item in sGD_index:
                sGD.append(varIDs[item])
            sGD = np.array(sGD)

            previous_weights = prev_weights
            j = sGD_index[0]

            # Calculate and find maximum LgM for Child Left Node (variant==0)
            current_node_variant_selected = variants[j].reshape(1, -1)
            current_node_weights00 = torch.ones(current_node_variant_selected.shape, dtype=torch.float64,
                                                device=device) - current_node_variant_selected
            weights00 = previous_weights * current_node_weights00
            length = trace + 2
            lgMv00 = DriverSearch(weights00, traits, variants, length, q, device)[0]
            new_score_0 = torch.max(lgMv00, dim=0).values.cpu().numpy()[0]

            # Calculate and find maximum LgM for Child right Node (variant==1)
            current_node_weights11 = current_node_variant_selected
            weights11 = previous_weights * current_node_weights11
            length = trace + 2
            lgMv11 = DriverSearch(weights11, traits, variants, length, q, device)[0]
            new_score_1 = torch.max(lgMv11, dim=0).values.cpu().numpy()[0]

            # New Score while split into two Branches
            New_score = new_score_0 + new_score_1

            # Condition to stop or continue
            if New_score > Best_score:
                trace = trace + 1
                collect = collect
                collect.append(trace)
                Best_variable = sGD[0]
                Best_variable_index = sGD_index[0]
                snp = [Best_variable]

                # Get corresponding odds ratio and 2 by 2 table
                OR_details = get_or_details(snp, x, j)
                _list = [Best_score, Best_variable, trace, left_or_right, OR_details,
                         New_score]
                pointer.append(_list)
                print("index:{},best_variable:{},new_score:{},best_score:{},new_score_0:{},new_score_1:{},length:{}".
                      format(left_or_right, Best_variable, New_score, Best_score, new_score_0, new_score_1,length))
                ptr0 = dt(Best_variable, new_score_0, trace, 0, weights00, root_lgM, OR_details, collect, pointer,
                          q, device)
                ptr1 = dt(Best_variable, new_score_1, trace, 1, weights11, root_lgM, OR_details, collect, pointer,
                          q, device)

            else:
                bv = 0
                try:
                    snp = [varID]
                    # Get corresponding odds ratios and 2 by 2 table
                    _details = get_or_details(snp, x, i)
                except:
                    _details = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                print("index:{},best_variable:{},new_score:{},best_score:{},new_score_0:{},new_score_1:{},length:{}".
                      format(left_or_right, bv, New_score, Best_score, new_score_0, new_score_1,length))
                _list = [Best_score, bv, trace, left_or_right, _details, New_score]
                pointer.append(_list)

    return pointer


if __name__ == '__main__':
    variant_input = os.path.join('x10.csv')  # variant data input
    trait_input = os.path.join('y10.csv')  # traits input
    parameter = 1  # Parameter tuning input

    subIDs, varIDs, variants, df_variants = read_variantsF1(variant_input)
    subIDs_BP, traitIDs, traits = read_traitsF(trait_input)
    SGA = pd.read_csv(variant_input, index_col=0)
    SGA = SGA.T
    final = []
    for i in SGA.index.tolist():
        j = int(i)
        final.append(j)
    SGA['index'] = final
    SGA = SGA.set_index('index')

    DEG_MPL = pd.read_csv(trait_input, index_col=0)
    DEG_MPL.columns = ['pheno']
    SGA_DEG = pd.concat([DEG_MPL, SGA], axis=1)

    clock_start = datetime.now()
    start = time.perf_counter()
    logging.info(clock_start)

    # Using For Loop to Generate n Bootstrap Datasets and Run dt functions
    for qq in tqdm(list(range(201))[1:201]):
        _len = SGA_DEG.shape[0] + 1
        SGA_DEG2 = SGA_DEG.copy()
        SGA_DEG2 = SGA_DEG2.sample(frac=1, replace=True, random_state=qq)
        SGA_DEG2['index'] = list(range(_len))[1:]
        SGA_DEG2 = SGA_DEG2.set_index('index')
        SGA_DEG3 = SGA_DEG2.copy()

        # Get varIDs, subIDs, variants, traits
        SGA_DEG3.insert(loc=1, column='A0', value=1)
        variants = SGA_DEG3.iloc[:, 1:].values.T
        traits = SGA_DEG3[["pheno"]].values
        varIDs = list(SGA_DEG3.columns[1:])
        subIDs = list(int(x) for x in SGA_DEG3.index)

        start_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Find Root Node
        traits = torch.as_tensor(traits, dtype=torch.float64, device=device)
        variants = torch.as_tensor(variants, dtype=torch.float64, device=device)
        rr, glgm, glgm_topGD, topGD_index = Finding_RootNode(traits, variants)
        var = varIDs[topGD_index[0]]

        # Find Root Node's LgM and Odds ratio
        sga_pheno = torch.as_tensor(SGA_DEG3['pheno'].values, dtype=torch.float64, device=device)
        sga_var = torch.as_tensor(SGA_DEG3[var].values, dtype=torch.float64, device=device)
        root_lgM, OR_details = root_node_function(sga_pheno, sga_var, var, variants, q=parameter, device=device)
        # logging.info("elpased time:{}".format(time.time() - start_time))

        # Run dt function
        element_run = dt(var, -math.inf, 0, 2, [], root_lgM, OR_details, collect=[], pointer=[], q=parameter,
                         device=device)
        logging.info("elpased time:{}".format(time.time() - start_time))
        for i in np.arange(len(element_run)):
            element_run[i].append(qq)
        if qq == 1:
            first_df = pd.DataFrame(element_run)
        else:
            second_df = pd.DataFrame(element_run)
            first_df = pd.concat([first_df, second_df], axis=0)

    # save data (Trees)
    columns = ['Marginal', 'Node', 'Level', 'Branch', 'OR_Details', 'Additional_Stat', 'Tree']
    first_df.columns = columns
    first_df.to_csv(os.path.join("IBIDT_trees.csv"))
    logging.info("total elapsed time:{}".format((datetime.now() - clock_start).seconds))

