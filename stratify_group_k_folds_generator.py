import pandas as pd
import numpy as np
def _iterate_over_df(df, folds, _sort_dict):
    """ Aid function to count how many goods/ bads for each group """
    for _, row in df.iterrows():
        fold = sorted(folds.items(), key = _sort_dict, reverse = False)[0][0]
        folds[fold]['group'].append(row['group'])
        folds[fold]['Negative']  += row['count']
        folds[fold]['Possitive'] += row['y']
    return folds

def stratify_group_k_folds_generator(y, groups, cv, random_seed = 14):
    """ 
    creating stratify folds with some group split uniquely to some group. Heuristically and will not always work optimally, 
    most of the times should perform very good.
    Logic:
    1. creating table of each fold, how many target units it has and many good units.
    2. creating fold dictionary, containing which groups it has and how many good/bad units it has
    3. sorting by amount of target units and then by good units
    4. iterating over the groups with target units, each iteration adding this group to the fold with the lowest amount of targets
    5. iterating over the groups without target units, each iteration adding this group to the fold with the lowest amount of good units 
       for balance
       
    Parameters
    ----------
        y: array like
            Some binary array, target array
        groups: array like
            Some array of values (can be anything) that folds should contain unique values 
        cv: int
            Number of folds
        random_seed: int
            Seed to set for reproducability
    """
    np.random.seed(random_seed)
    # creating dicionary of folds, step 2
    folds = {i:{'Negative':0, 'Possitive':0, 'group':[]} for i in range(cv)}
    # grouping
    groups = pd.Series(groups).astype('category').cat.codes.values
    temp = pd.DataFrame({'group':groups, 'y':np.array(y)}).groupby('group',as_index = False).y.agg({'y' : np.sum, 'count' : 'count'})\
                        .sort_values(['y','count'], ascending = False)
    folds = _iterate_over_df(temp[temp.y > 0],  folds, lambda x: x[1]['Possitive'])
    folds = _iterate_over_df(temp[temp.y == 0], folds, lambda x: x[1]['Negative'])
    for i in range(cv):
        test_groups = folds[i]['group']
        yield np.where(~np.isin(groups, test_groups))[0], np.where(np.isin(groups, test_groups))[0]
        