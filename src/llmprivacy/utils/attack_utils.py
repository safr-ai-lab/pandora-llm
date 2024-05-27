import torch

# TODO processing should not occur here
def split_pt_into_dict(pt_file, only_x=False, only_theta=False, divideby = 10000000):
    '''
    Convert .pt of norms into dictionary. Divide L1 norms by divideby 
    for numerical overflow issues. 
    '''    
    train_stat, val_stat = torch.load(pt_file)
    tstat_dict = {}
    valstat_dict = {}
    if only_x or (not only_x and not only_theta):
        tstat_dict["linf_x"] = train_stat[:,0]
        tstat_dict["l1_x"] = train_stat[:,1]
        tstat_dict["l2_x"] = train_stat[:,2]
        valstat_dict["linf_x"] = val_stat[:,0]
        valstat_dict["l1_x"] = val_stat[:,1]
        valstat_dict["l2_x"] = val_stat[:,2]

    #separators
    s1 = 6 
    total_vector_len = train_stat.shape[1]
    s2 = s1+(train_stat.shape[1]-s1) // 3
    s3 = s1+((train_stat.shape[1]-s1) // 3) * 2
    s4 = total_vector_len

    if only_theta or (not only_x and not only_theta):
        tstat_dict["linf_layers"] = train_stat[:,s1:s2]
        tstat_dict["l1_layers"] = train_stat[:,s2:s3]
        tstat_dict["l2_layers"] = train_stat[:,s3:s4]

        tstat_dict["linf_theta"] = tstat_dict["linf_layers"].abs().max(dim=1).values
        tstat_dict["l1_theta"] = (tstat_dict["l1_layers"]/divideby).abs().sum(axis=1)
        tstat_dict["l2_theta"] = (tstat_dict["l2_layers"]**2).sum(dim=1)

        valstat_dict["linf_layers"] = val_stat[:,s1:s2]
        valstat_dict["l1_layers"] = val_stat[:,s2:s3]
        valstat_dict["l2_layers"] = val_stat[:,s3:s4]

        valstat_dict["linf_theta"] = valstat_dict["linf_layers"].abs().max(dim=1).values
        valstat_dict["l1_theta"] = (valstat_dict["l1_layers"]/divideby).abs().sum(axis=1)
        valstat_dict["l2_theta"] = (valstat_dict["l2_layers"]**2).sum(dim=1)


    return tstat_dict, valstat_dict

def split_unsplit_pt(pt_file):
    '''
    Convert pt_file into dictionary and then .pt file
    '''
    train_stat, val_stat = split_pt_into_dict(pt_file)
    return torch.cat(list(train_stat.values()),axis=0), torch.cat(list(val_stat.values()),axis=0)