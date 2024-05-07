import sys, os
import matplotlib.pyplot as plt
import dit
import numpy as np

from FACT.helper import *
from FACT.fairness import *
from FACT.data_util import *
from FACT.plot import *
from FACT.lin_opt import *
from sklearn.utils import shuffle





def shuffle_clients_data(clients_data, p):
    def shuffle_arrays(arrays, indices):
        """Shuffles multiple arrays with the same indices."""
        return [array[indices] for array in arrays]

    # Get data from clients
    client_0_data = clients_data['client_0']
    client_1_data = clients_data['client_1']

    # Determine the number of data points to shuffle
    num_data_points_0_train = len(client_0_data['X_train'])
    num_data_points_1_train = len(client_1_data['X_train'])
    num_data_points_0_test = len(client_0_data['X_test'])
    num_data_points_1_test = len(client_1_data['X_test'])

    num_to_shuffle_train = int(min(num_data_points_0_train, num_data_points_1_train) * p)
    num_to_shuffle_test = int(min(num_data_points_0_test, num_data_points_1_test) * p)

    # Generate shuffle indices for both clients
    shuffle_indices_0_train = np.random.permutation(num_data_points_0_train)[:num_to_shuffle_train]
    shuffle_indices_1_train = np.random.permutation(num_data_points_1_train)[:num_to_shuffle_train]
    shuffle_indices_0_test = np.random.permutation(num_data_points_0_test)[:num_to_shuffle_test]
    shuffle_indices_1_test = np.random.permutation(num_data_points_1_test)[:num_to_shuffle_test]

    # Extract data to be shuffled from both clients for training and test datasets
    keys_train = ['X_train', 'y_train', 'X_train_removed']
    keys_test = ['X_test', 'y_test', 'X_test_removed']

    data_0_to_shuffle_train = [client_0_data[key][shuffle_indices_0_train] for key in keys_train]
    data_1_to_shuffle_train = [client_1_data[key][shuffle_indices_1_train] for key in keys_train]
    data_0_to_shuffle_test = [client_0_data[key][shuffle_indices_0_test] for key in keys_test]
    data_1_to_shuffle_test = [client_1_data[key][shuffle_indices_1_test] for key in keys_test]

    # Swap data between clients
    for i, key in enumerate(keys_train):
        client_0_data[key][shuffle_indices_0_train] = data_1_to_shuffle_train[i]
        client_1_data[key][shuffle_indices_1_train] = data_0_to_shuffle_train[i]

    for i, key in enumerate(keys_test):
        client_0_data[key][shuffle_indices_0_test] = data_1_to_shuffle_test[i]
        client_1_data[key][shuffle_indices_1_test] = data_0_to_shuffle_test[i]

    return clients_data





# split_method=='iid' Scenario 1
# split_method=='heterogeneity' Scenario 2
# split_method=='synergy' Scenario 3

def split_client_data(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, client_no=2, sex_idx = 2, split_method='iid'):

    

    if split_method=='iid':
        clients_data = {}


        X_train, y_train, X_train_removed = shuffle(X_train, y_train, X_train_removed, random_state=42)
        X_test, y_test, X_test_removed = shuffle(X_test, y_test, X_test_removed, random_state=42)
        
        # Split the data across clients
        X_train = np.split(X_train, client_no)
        y_train = np.split(y_train, client_no)
        X_test = np.split(X_test, client_no)
        y_test = np.split(y_test, client_no)
        X_train_removed = np.split(X_train_removed, client_no)
        X_test_removed = np.split(X_test_removed, client_no)

        for i in range(client_no):
            client_key = f"client_{i}"
            client_data = {'X_train': X_train[i], 'y_train': y_train[i],'X_test': X_test[i], 'y_test': y_test[i],'X_train_removed': X_train_removed[i], 'X_test_removed': X_test_removed[i] }
            clients_data[client_key] = client_data

    if split_method=='heterogeneity':
        clients_data = {} # alpha = 0.85

        # Separate data based on the sensitive attribute
        male_indices_train = X_train[:, sex_idx] == 1
        female_indices_train = X_train[:, sex_idx] == 0

        male_indices_test = X_test[:, sex_idx] == 1
        female_indices_test = X_test[:, sex_idx] == 0

        # Get data points corresponding to males and females
        X_train_male, y_train_male = X_train[male_indices_train], y_train[male_indices_train]
        X_train_female, y_train_female = X_train[female_indices_train], y_train[female_indices_train]

        X_test_male, y_test_male = X_test[male_indices_test], y_test[male_indices_test]
        X_test_female, y_test_female = X_test[female_indices_test], y_test[female_indices_test]

        # Get 'removed' versions without the sensitive attribute
        X_train_removed_male = X_train_removed[male_indices_train]
        X_train_removed_female = X_train_removed[female_indices_train]
        X_test_removed_male = X_test_removed[male_indices_test]
        X_test_removed_female = X_test_removed[female_indices_test]

        # Shuffle data
        X_train_male, y_train_male, X_train_removed_male = shuffle(X_train_male, y_train_male, X_train_removed_male, random_state=42)
        X_train_female, y_train_female, X_train_removed_female = shuffle(X_train_female, y_train_female, X_train_removed_female, random_state=42)

        X_test_male, y_test_male, X_test_removed_male = shuffle(X_test_male, y_test_male, X_test_removed_male, random_state=42)
        X_test_female, y_test_female, X_test_removed_female = shuffle(X_test_female, y_test_female, X_test_removed_female, random_state=42)

        # Get 90% of male data for client 1 and 10% for client 2, and vice versa for female data
        split_ratio = 0.92
        clients_data['client_0'] = {
            'X_train': np.vstack((X_train_male[:int(split_ratio*len(X_train_male))], X_train_female[:int((1-split_ratio)*len(X_train_female))])),
            'y_train': np.concatenate((y_train_male[:int(split_ratio*len(y_train_male))], y_train_female[:int((1-split_ratio)*len(y_train_female))])),
            'X_test': np.vstack((X_test_male[:int(split_ratio*len(X_test_male))], X_test_female[:int((1-split_ratio)*len(X_test_female))])),
            'y_test': np.concatenate((y_test_male[:int(split_ratio*len(y_test_male))], y_test_female[:int((1-split_ratio)*len(y_test_female))])),
            'X_train_removed': np.vstack((X_train_removed_male[:int(split_ratio*len(X_train_removed_male))], X_train_removed_female[:int((1-split_ratio)*len(X_train_removed_female))])),
            'X_test_removed': np.vstack((X_test_removed_male[:int(split_ratio*len(X_test_removed_male))], X_test_removed_female[:int((1-split_ratio)*len(X_test_removed_female))])),
        }

        clients_data['client_1'] = {
            'X_train': np.vstack((X_train_male[int(split_ratio*len(X_train_male)):], X_train_female[int((1-split_ratio)*len(X_train_female)):])),
            'y_train': np.concatenate((y_train_male[int(split_ratio*len(y_train_male)):], y_train_female[int((1-split_ratio)*len(y_train_female)):])),
            'X_test': np.vstack((X_test_male[int(split_ratio*len(X_test_male)):], X_test_female[int((1-split_ratio)*len(X_test_female)):])),
            'y_test': np.concatenate((y_test_male[int(split_ratio*len(y_test_male)):], y_test_female[int((1-split_ratio)*len(y_test_female)):])),
            'X_train_removed': np.vstack((X_train_removed_male[int(split_ratio*len(X_train_removed_male)):], X_train_removed_female[int((1-split_ratio)*len(X_train_removed_female)):])),
            'X_test_removed': np.vstack((X_test_removed_male[int(split_ratio*len(X_test_removed_male)):], X_test_removed_female[int((1-split_ratio)*len(X_test_removed_female)):])),
        }

    if split_method=='synergy':
        clients_data = {}

        # Create masks for each condition
        condition_00 = (X_train[:, sex_idx] == 0) & (y_train == 0)
        condition_11 = (X_train[:, sex_idx] == 1) & (y_train == 1)
        condition_01 = (X_train[:, sex_idx] == 0) & (y_train == 1)
        condition_10 = (X_train[:, sex_idx] == 1) & (y_train == 0)

        # Separate data based on the conditions
        X_train_00, y_train_00 = X_train[condition_00], y_train[condition_00]
        X_train_11, y_train_11 = X_train[condition_11], y_train[condition_11]
        X_train_01, y_train_01 = X_train[condition_01], y_train[condition_01]
        X_train_10, y_train_10 = X_train[condition_10], y_train[condition_10]

        # Separate the 'removed' datasets in the same manner
        X_train_removed_00 = X_train_removed[condition_00]
        X_train_removed_11 = X_train_removed[condition_11]
        X_train_removed_01 = X_train_removed[condition_01]
        X_train_removed_10 = X_train_removed[condition_10]

        # Do the same for the test dataset
        condition_00_test = (X_test[:, sex_idx] == 0) & (y_test == 0)
        condition_11_test = (X_test[:, sex_idx] == 1) & (y_test == 1)
        condition_01_test = (X_test[:, sex_idx] == 0) & (y_test == 1)
        condition_10_test = (X_test[:, sex_idx] == 1) & (y_test == 0)

        X_test_00, y_test_00 = X_test[condition_00_test], y_test[condition_00_test]
        X_test_11, y_test_11 = X_test[condition_11_test], y_test[condition_11_test]
        X_test_01, y_test_01 = X_test[condition_01_test], y_test[condition_01_test]
        X_test_10, y_test_10 = X_test[condition_10_test], y_test[condition_10_test]

        # Separate the 'removed' test datasets in the same manner
        X_test_removed_00 = X_test_removed[condition_00_test]
        X_test_removed_11 = X_test_removed[condition_11_test]
        X_test_removed_01 = X_test_removed[condition_01_test]
        X_test_removed_10 = X_test_removed[condition_10_test]

        # Assign data to clients
        clients_data['client_0'] = {
            'X_train': np.vstack((X_train_00, X_train_11)),
            'y_train': np.concatenate((y_train_00, y_train_11)),
            'X_test': np.vstack((X_test_00, X_test_11)),
            'y_test': np.concatenate((y_test_00, y_test_11)),
            'X_train_removed': np.vstack((X_train_removed_00, X_train_removed_11)),
            'X_test_removed': np.vstack((X_test_removed_00, X_test_removed_11)),
        }

        clients_data['client_1'] = {
            'X_train': np.vstack((X_train_01, X_train_10)),
            'y_train': np.concatenate((y_train_01, y_train_10)),
            'X_test': np.vstack((X_test_01, X_test_10)),
            'y_test': np.concatenate((y_test_01, y_test_10)),
            'X_train_removed': np.vstack((X_train_removed_01, X_train_removed_10)),
            'X_test_removed': np.vstack((X_test_removed_01, X_test_removed_10)),
        }

    clients_data = shuffle_clients_data(clients_data, 0.01)
    return clients_data




def FairnessMeasures_Fl(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, client_data, model, sex_idx=2):
    #global fairness measures
    gfm = FairnessMeasures(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, model, sex_idx)
    cfm = {}

    for i in range(len(client_data)):

        key = f"client_{i}"
        cfm[key] = FairnessMeasures(
            client_data[key]['X_train'], 
            client_data[key]['y_train'], 
            client_data[key]['X_test'], 
            client_data[key]['y_test'], 
            client_data[key]['X_train_removed'], 
            client_data[key]['X_test_removed'], 
            model, sex_idx
        )

    return gfm, cfm 



def get_fairness_mats_FL(gfm, cfm,  verbose=False):
    """
    :param fm: FairnessMeasure object
    :return: a dictionary of fairness matrices, and constraint matrices/vectors
    """
    num_clients = len(cfm)

    N1 = gfm.pos_group.shape[0]
    N0 = gfm.neg_group.shape[0]

    N01 = cfm['client_0'].pos_group.shape[0]
    N00 = cfm['client_0'].neg_group.shape[0]
    
    N11 = cfm['client_1'].pos_group.shape[0]
    N10 = cfm['client_1'].neg_group.shape[0]


    mu1_idx = list(set(gfm.pos_gt).intersection(set(gfm.pos_group)))
    mu1 = len(mu1_idx)
    mu0_idx = list(set(gfm.pos_gt).intersection(set(gfm.neg_group)))
    mu0 = len(mu0_idx)

    mu01_idx = list(set(cfm['client_0'].pos_gt).intersection(set(cfm['client_0'].pos_group)))
    mu01 = len(mu01_idx)
    mu00_idx = list(set(cfm['client_0'].pos_gt).intersection(set(cfm['client_0'].neg_group)))
    mu00 = len(mu00_idx)

    mu11_idx = list(set(cfm['client_1'].pos_gt).intersection(set(cfm['client_1'].pos_group)))
    mu11 = len(mu11_idx)
    mu10_idx = list(set(cfm['client_1'].pos_gt).intersection(set(cfm['client_1'].neg_group)))
    mu10 = len(mu10_idx)

    assert N00 + N10 == N0, "The condition N00 + N10 != N0 is not met."
    assert N01 + N11 == N1, "The condition N01 + N11 != N1 is not met."
    # Additional conditions for mu0 and mu1 based on your requirements
    assert mu00+mu10 == mu0, "The condition for mu0 is not met."
    assert mu01+mu11 == mu1, "The condition for mu1 is not met."

    N = N1 + N0

    # Natural Constraints
    M = np.array([[1,1,1,1], [1,1,0,0]])
    M_const = np.zeros((8, 16))
    M_const[:2, :4] = M 
    M_const[2:4, 4:8] = M
    M_const[4:6, 8:12] = M
    M_const[6:, 12:] = M 
    b_const = np.array([[N11, mu11, N10, mu10, N01, mu01, N00, mu00]]).T

    z = np.array([
        cfm['client_1'].pos_group_stats['TP'], cfm['client_1'].pos_group_stats['FN'], cfm['client_1'].pos_group_stats['FP'], cfm['client_1'].pos_group_stats['TN'],
        cfm['client_1'].neg_group_stats['TP'], cfm['client_1'].neg_group_stats['FN'], cfm['client_1'].neg_group_stats['FP'], cfm['client_1'].neg_group_stats['TN'],
        cfm['client_0'].pos_group_stats['TP'], cfm['client_0'].pos_group_stats['FN'], cfm['client_0'].pos_group_stats['FP'], cfm['client_0'].pos_group_stats['TN'],
        cfm['client_0'].neg_group_stats['TP'], cfm['client_0'].neg_group_stats['FN'], cfm['client_0'].neg_group_stats['FP'], cfm['client_0'].neg_group_stats['TN']
    ]).T
    assert np.array_equal(np.dot(M_const, z).flatten(), b_const.flatten()), "The condition M_const * z != b_const is not met."
    # Global Demographic ParityM


    M_dp = np.array([[N0, 0, N0, 0, -N1, 0, -N1, 0, N0, 0, N0, 0, -N1, 0, -N1, 0]])/N
    b_dp = np.zeros((1,1))
    aa,bb = gfm.group_parity()
       # Local  Demographic ParityM


    M_l = np.array([[N10, 0, N10, 0, -N11, 0, -N11, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, N00, 0, N00, 0, -N01, 0, -N01, 0]])/N

    b_l = np.zeros((2,1))
    # NOTE Normalization
    b_const = 1 / N * b_const
    measures = [(M_dp, b_dp, 'global_DP'),
                (M_l, b_l, 'local_DP')]

    measures_dict = dict()
    for m in measures:
        measures_dict[m[2]] = (m[0], m[1])
    return measures, measures_dict, M_const, b_const

def test_fair_instance_FL(gfm, cfm, names, opt_target='performance', eps=0, err_ub=0.1, lmbd=1, epsdelta=True):
    """
    Test the solution of fairness linear program
    :param fm:  FairnessMeasure object
    :param names:  measures of fairness to analyze
    :param opt_target: optimizing target. can be either performance or fairness.
    :param eps: relaxation of the fairness equality constraint when optimizing performance
    :param err_ub: the upper bound of the error rate for performance constraint when optimizing fairness
    :param lmbd: (only for joint optimization) regularization constant
    :return: optimization results
    """
    # Get matrices
    mat, mat_dict, M_const, b_const = get_fairness_mats_FL(gfm, cfm)
    try:
        Ms = [mat_dict[n][0] for n in names]
        bs = [mat_dict[n][1] for n in names]
    except KeyError:
        raise ValueError("The provided names are not supported")
    M = np.concatenate(Ms, axis=0)
    b = np.concatenate(bs, axis=0)

    # set values for c to pick up the mistakes (FP/FN) in the solution vector
    d = M.shape[1]
    c = np.zeros(d)
    tmp = int(d / 4)
    pos_idx = []
    neg_idx = []
    for i in range(tmp):
        pos_idx = pos_idx + [i*4, i*4+3]
        neg_idx = neg_idx + [i*4+1, i*4+2]
    c[pos_idx] = 0
    c[neg_idx] = 1
    # print(f'c={c}')

    if opt_target == 'performance':
        # optimize performance with fairness constraints
        #res = opt_perf_const_fairness(M, b, c, M_const, b_const, eps=eps)
        res = eps_opt(M, c, M_const, b_const, eps=eps, seed=0, method='SLSQP', epsdelta=epsdelta)
    elif opt_target == 'fairness':
        # optimize fairness with performance constraints
        res =  opt_fairness_const_perf(M, b, c, M_const, b_const, err_ub=err_ub)
    elif opt_target == 'joint':
        # optimize regularized version
        res = solve_LAFOP(M, M_const, b_const, lmbd=lmbd, seed=0)
    else:
        raise ValueError('Target not supported.')

    return res


def check_fairness_solution(z, mats_dict):

    global_val = np.dot(mats_dict['global_DP'][0], z)
    local_val  = np.dot(mats_dict['local_DP'][0], z)

    return global_val, local_val



def PID(B):
    A=['100','101','110','111','000','001','010','011']
    print(A,B)

    fd=dit.Distribution(A,B)
    fd.set_rv_names('SZY')
    fd_pid = dit.pid.PID_BROJA(fd, ['S', 'Y'], 'Z')
    #print(fd_pid)
    Uniq=fd_pid._pis[(('Y',),)]
    Red = fd_pid._pis[(('S',), ('Y',))]
    Syn = fd_pid._pis[(('S', 'Y'),)]
    Uniq_S = fd_pid._pis[(('S',),)]
    I_YS=dit.shannon.mutual_information(fd, ['Y'], ['S'])
    I_ZS=dit.shannon.mutual_information(fd, ['Z'], ['S'])
    I_ZY=dit.shannon.mutual_information(fd, ['Z'], ['Y'])
    I_ZY_S= dit.multivariate.coinformation(fd, 'ZY', 'S')
    CoI= dit.multivariate.coinformation(fd, ['Y', 'Z', 'S'])
    return Uniq, Red, Syn ,Uniq_S, I_ZS, I_YS, I_ZY, I_ZY_S , CoI



def P_SZY(z):

    p100 = np.dot(z, np.array([0,0,0,0   ,0,1,0,1   ,0,0,0,0   ,0,0,0,0]))
    p101 = np.dot(z, np.array([0,0,0,0   ,1,0,1,0   ,0,0,0,0   ,0,0,0,0]))
    p110 = np.dot(z, np.array([0,1,0,1   ,0,0,0,0   ,0,0,0,0   ,0,0,0,0]))
    p111 = np.dot(z, np.array([1,0,1,0   ,0,0,0,0   ,0,0,0,0   ,0,0,0,0]))
    p000 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,0,0,0   ,0,1,0,1]))
    p001 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,0,0,0   ,1,0,1,0]))
    p010 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,1,0,1   ,0,0,0,0]))
    p011 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,1,0,1,0   ,0,0,0,0]))

    dist = np.array([p100, p101, p110, p111, p000, p001, p010, p011])

    assert np.isclose(np.sum(dist), 1, atol=1e-9), f"The condition sum(dist) != 1 is not met. sum = {np.sum(dist)}"

    
    return dist.tolist()



def measures(z): 

    p100 = np.dot(z, np.array([0,0,0,0   ,0,1,0,1   ,0,0,0,0   ,0,0,0,0]))
    p101 = np.dot(z, np.array([0,0,0,0   ,1,0,1,0   ,0,0,0,0   ,0,0,0,0]))
    p110 = np.dot(z, np.array([0,1,0,1   ,0,0,0,0   ,0,0,0,0   ,0,0,0,0]))
    p111 = np.dot(z, np.array([1,0,1,0   ,0,0,0,0   ,0,0,0,0   ,0,0,0,0]))
    p000 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,0,0,0   ,0,1,0,1]))
    p001 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,0,0,0   ,1,0,1,0]))
    p010 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,0,1,0,1   ,0,0,0,0]))
    p011 = np.dot(z, np.array([0,0,0,0   ,0,0,0,0   ,1,0,1,0   ,0,0,0,0]))

    dist = np.array([p100, p101, p110, p111, p000, p001, p010, p011])
    dist = dist/ np.sum(dist)

    assert np.isclose(np.sum(dist), 1, atol=1e-9), "The condition sum(dist) != 1 is not met."

    A=['100','101','110','111','000','001','010','011']

    B = dist.tolist()

    fd=dit.Distribution(A,B)
    fd.set_rv_names('SZY')
    fd_pid = dit.pid.PID_BROJA(fd, ['S', 'Y'], 'Z')
    #print(fd_pid)
    Uniq=fd_pid._pis[(('Y',),)]
    Red = fd_pid._pis[(('S',), ('Y',))]
    Syn = fd_pid._pis[(('S', 'Y'),)]
    Uniq_S = fd_pid._pis[(('S',),)]
    I_YS=dit.shannon.mutual_information(fd, ['Y'], ['S'])
    I_ZS=dit.shannon.mutual_information(fd, ['Z'], ['S'])
    I_ZY=dit.shannon.mutual_information(fd, ['Z'], ['Y'])
    I_ZY_S= dit.multivariate.coinformation(fd, 'ZY', 'S')
    CoI= dit.multivariate.coinformation(fd, ['Y', 'Z', 'S'])


    # Global and Client Statistical Parity 

    Glb_SP = ((p111 + p011)/ (p111 + p011 + p110+p010)) - (p101+p001)/(p101 + p001 + p000 + p100)

    loc_SP1 = (p111/(p110 + p111)) - (p101/(p101+p100))
               
    loc_SP0 =  (p011/(p010 + p011)) - (p001/(p001+p000)) 


    return {'Uniq': Uniq, 'Red': Red,'Syn': Syn, 'Uniq_S': Uniq_S, 'I_ZY': I_ZY, 'I_ZY_S': I_ZY_S, 'I_ZS': I_ZS, 'I_YS': I_YS, 'CoI': CoI, 'dist': dist, 'Glb_SP': Glb_SP, 'loc_SP1': loc_SP1, 'loc_SP0': loc_SP0}


def optimization_FL(gfm, cfm, opt_target='performance', eps_g=0, eps_l=0, err_ub=0.1, lmbd=1, epsdelta=True, seed=0):

    # Get matrices
    mat, mat_dict, M_const, b_const = get_fairness_mats_FL(gfm, cfm)
    

    # set values for c to pick up the mistakes (FP/FN) in the solution vector
    d = M_const.shape[1]

    c = np.zeros(d)
    tmp = int(d / 4)
    pos_idx = []
    neg_idx = []
    for i in range(tmp):
        pos_idx = pos_idx + [i*4, i*4+3]
        neg_idx = neg_idx + [i*4+1, i*4+2]
    c[pos_idx] = 0
    c[neg_idx] = 1
    # print(f'c={c}')

    if opt_target == 'performance':
      
        def ineq_con(x):
            return eps_g - measures(x)['I_ZY']
       
        def ineq_con_loc(x):
            return eps_l - measures(x)['I_ZY_S']
        def eq_con(x):
            return np.dot(M_const, x) - np.squeeze(b_const)

        
        cons = [{'type':'eq', 'fun':eq_con}, {'type':'ineq', 'fun':ineq_con}, {'type':'ineq', 'fun':ineq_con_loc}]

        def obj(x):
            return np.dot(c, x)


        np.random.seed(seed)
        x0 = np.random.random(d)
        bounds = [(0,1) for _ in range(d)]
        res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return res
    elif opt_target == 'fairness':

        res =  opt_fairness_const_perf(M, b, c, M_const, b_const, err_ub=err_ub)
    elif opt_target == 'joint':

        res = solve_LAFOP(M, M_const, b_const, lmbd=lmbd, seed=0)
    else:
        raise ValueError('Target not supported.')

    return res



def optimization_FL_MS(gfm, cfm, opt_target='performance', eps_g=0, eps_l=0, err_ub=0.1, lmbd=1, epsdelta=True, seed=0):

    # Get matrices
    mat, mat_dict, M_const, b_const = get_fairness_mats_FL(gfm, cfm)

    d = M_const.shape[1]

    c = np.zeros(d)
    tmp = int(d / 4)
    pos_idx = []
    neg_idx = []
    for i in range(tmp):
        pos_idx = pos_idx + [i*4, i*4+3]
        neg_idx = neg_idx + [i*4+1, i*4+2]
    c[pos_idx] = 0
    c[neg_idx] = 1
    # print(f'c={c}')



    fct_model = np.array([gfm.pos_group_stats['TP'],
                                gfm.pos_group_stats['FN'], 
                                gfm.pos_group_stats['FP'], 
                                gfm.pos_group_stats['TN'],
                                gfm.neg_group_stats['TP'], 
                                gfm.neg_group_stats['FN'], 
                                gfm.neg_group_stats['FP'], 
                                gfm.neg_group_stats['TN']])
    fct_model = fct_model / np.sum(fct_model)

    N = gfm.y_test.shape[0]
    M_pos = gfm.mu_pos / N
    NM_pos = gfm.pos_group_num / N - M_pos
    M_neg = gfm.mu_neg / N
    NM_neg = gfm.neg_group_num /N - M_neg

           

    fpr_pos_1 = max(fct_model[2] / NM_pos, fct_model[3] / NM_pos)
    fpr_pos_2 = 1 - fpr_pos_1
    tpr_pos_1 = min(fct_model[0] / M_pos, fct_model[1] / M_pos)
    tpr_pos_2 = 1 - tpr_pos_1
    fpr_neg_1 = max(fct_model[6] / NM_neg, fct_model[7] / NM_neg)
    fpr_neg_2 = 1 - fpr_neg_1
    tpr_neg_1 = min(fct_model[4] / M_neg, fct_model[5] / M_neg)
    tpr_neg_2 = 1 - tpr_neg_1




    if opt_target == 'performance':
        def MS_con1(x):

            fpr_x_pos = (x[2]+x[10]) / NM_pos                                                                                                                                                                                                                                     
            fpr_x_neg = (x[6]+x[14]) / NM_neg
            tpr_x_pos = (x[0]+x[8]) / M_pos
            tpr_x_neg = (x[4]+x[12]) / M_neg
            return tpr_pos_2 / fpr_pos_2 * fpr_x_pos - tpr_x_pos

        def MS_con2(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_x_pos - tpr_pos_1 / tpr_pos_1 * fpr_x_pos

        def MS_con3(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_pos_1 / fpr_pos_1 * (fpr_x_pos - 1) + 1 - tpr_x_pos

        def MS_con4(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_x_pos - tpr_pos_2 / fpr_pos_2 * (fpr_x_pos - 1) - 1

        def MS_con5(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_neg_2 / fpr_neg_2 * fpr_x_neg - tpr_x_neg

        def MS_con6(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_x_neg - tpr_neg_1 / tpr_neg_1 * fpr_x_neg

        def MS_con7(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_neg_1 / fpr_neg_1 * (fpr_x_neg - 1) + 1 - tpr_x_neg

        def MS_con8(x):
            fpr_x_pos = (x[2] + x[10]) / NM_pos
            fpr_x_neg = (x[6] + x[14]) / NM_neg
            tpr_x_pos = (x[0] + x[8]) / M_pos
            tpr_x_neg = (x[4] + x[12]) / M_neg
            return tpr_x_neg - tpr_neg_2 / fpr_neg_2 * (fpr_x_neg - 1) - 1


        def ineq_con(x):
            return eps_g - measures(x)['I_ZY']
       
        def ineq_con_loc(x):
            return eps_l - measures(x)['I_ZY_S']
        def eq_con(x):
            return np.dot(M_const, x) - np.squeeze(b_const)

        cons = [
                {'type':'eq', 'fun':eq_con},
                {'type':'ineq', 'fun':ineq_con},
                {'type':'ineq', 'fun':ineq_con_loc},
                {'type':'ineq', 'fun':MS_con1},
                {'type':'ineq', 'fun':MS_con2},
                {'type':'ineq', 'fun':MS_con3},
                {'type':'ineq', 'fun':MS_con4},
                {'type':'ineq', 'fun':MS_con5},
                {'type':'ineq', 'fun':MS_con6},
                {'type':'ineq', 'fun':MS_con7},
                {'type':'ineq', 'fun':MS_con8},
            ]

        def obj(x):
            return np.dot(c, x)

        np.random.seed(seed)
        x0 = np.random.random(d)
        bounds = [(0,1) for _ in range(d)]
        res = optimize.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
        return res
    elif opt_target == 'fairness':
        res =  opt_fairness_const_perf(M, b, c, M_const, b_const, err_ub=err_ub)
    elif opt_target == 'joint':
        # optimize regularized version
        res = solve_LAFOP(M, M_const, b_const, lmbd=lmbd, seed=0)
    else:
        raise ValueError('Target not supported.')

    return res