#!/usr/bin/env python
# coding: utf-8
# run_hcc.py
# written by André Carrington
#
#    Copyright 2022 University of Ottawa
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

from CAE2Db import CAE2Db
import numpy as np

# HCC project
projectFolder = '/Users/acfakepath/hcc/'
cases         = ['78_1', '2_1', '2_2', '25_1', '29_1', '31_1', '12_1', '19_1']  # sample of studies from the dataset
imageDim      = (768, 1024)                      # y, x
cases2        = ['79_1', '22_1', '30_1', '66_1', '67_1', '73_1', '73_2', '73_3', '38_1']
imageDim2     = (720, 960)                       # y, x
labels        = [     1,     0,      1,     0,      1,      0,      1,      0]  
labels2       = [     1,     0,      1,     0,      1,      0,      1,      0,      1  ]
slices        = 1
liverMask     = lambda x: np.logical_or(x == 1, x == 2)
#lesionMask   = lambda x: (x == 2)
experiment    = dict(folds=2, repeats=1, splitterType='KFold')

# for HCC the ultrasound images have values in [0, 255]
files         = []
segmentations = []
for case in cases:
   files         = files         + [f'{projectFolder}{case}.dcm']
   segmentations = segmentations + [f'{projectFolder}{case}.nii.gz']
#endfor

hcc_CAE2D     = CAE2Db(projectfolder=projectFolder, cases=cases, files=files, segmentations=segmentations,
                      segmask=liverMask, labels=labels, imagedim=imageDim, slices=slices, experiment=experiment)
