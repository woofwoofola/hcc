#!/usr/bin/env python
# coding: utf-8
# splitDataNd.py
# written by André Carrington
#
# Copyright 2021 Ottawa Hospital Research Institute
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

def getTwoSplitters(splitterType, folds, repeats):
    split_random_state = 19
    if   splitterType == 'KFold':
        from sklearn.model_selection import KFold
        kf  = KFold(n_splits=folds, random_state=split_random_state,   shuffle=True)
        kf2 = KFold(n_splits=folds, random_state=split_random_state+1, shuffle=True)
        total_folds = folds
    elif splitterType == 'RepeatedKFold':
        from sklearn.model_selection import RepeatedKFold
        # shuffle=True is automatic and is not an option to specify
        kf  = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=split_random_state)
        kf2 = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=split_random_state+1)
        total_folds = folds * repeats
    elif splitterType == 'StratifiedKFold':
        from sklearn.model_selection import StratifiedKFold
        kf  = StratifiedKFold(n_splits=folds, random_state=split_random_state,   shuffle=True)
        kf2 = StratifiedKFold(n_splits=folds, random_state=split_random_state+1, shuffle=True)
        total_folds = folds
    elif splitterType == 'RepeatedStratifiedKFold':
        from sklearn.model_selection import RepeatedStratifiedKFold
        # shuffle=True is automatic and is not an option to specify
        kf  = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=split_random_state)
        kf2 = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=split_random_state+1)
        total_folds = folds * repeats
    else:
        ValueError('splitterType not recognized')
    #endif
    return total_folds, kf, kf2
#enddef

def getTrainAndValidationFoldIndices(y, splitterType, folds, repeats):
    ''' Gets fold data for training and validation and ensures both classes are present 
        and does not allow any validation fold to have only one class. To enforce both
        classes, a main and backup splitter are used, where the latter splitter is used if
        the requirement is not met by a fold in the main splitter.
        
        splitterType = {'KFold', 'RepeatedKFold', 'StratifiedKFold', 'RepeatedStratifiedKFold'}
    ''' 
    import numpy as np

    print(f'Making training/validation sets with {splitterType} cross validation.')
    total_folds, kf, kf2 = getTwoSplitters(splitterType, folds, repeats)

    dummy_x = np.transpose(np.array(y,y))

    tindex = [None] * total_folds
    vindex = [None] * total_folds

    i = 0  # index
    j = 0  # backup batch index
    # fix undocumented bug in sklearn's StratifiedKFold: include y in call to split
    for train_index, val_index in kf.split(dummy_x, y):
        # if the validation set has more than one class, then things are normal
        if len(np.unique(y.iloc[val_index])) > 1:
            tindex[i]     = train_index
            vindex[i]     = val_index
            i = i + 1
        else:
            # the validation set only had one class, so let's get indices
            # from the backup batch we hope are better, starting at index j==0 (to avoid reuse/duplication)
            k = 0
            for train_index2, val_index2 in kf2.split(dummy_x, y):
                if k < j:  # if not at starting index j, then loop to get there
                    k = k + 1
                    continue
                else:
                    # if the validation set has more than one class
                    if len(np.unique(y.iloc[val_index2])) > 1:
                        tindex[i]     = train_index2
                        vindex[i]     = val_index2
                        i = i + 1
                        j = j + 1
                        break
                    else:
                        continue
                    #endif
                #endif
            #endfor
        #endif
    #endfor
    return tindex, vindex
#enddef
