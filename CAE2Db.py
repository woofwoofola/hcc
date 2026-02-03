#!/usr/bin/env python
# -*- coding: latin-1 -*-
# CAE2db.py
# Written by André Carrington
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

class CAE2Db(object):

    def __init__(self, projectfolder, cases, files, segmentations, segmask, labels, imagedim, slices,
                 experiment, quiet=False):
        '''CAE2D constructor'''
        import transcript
        import time

        if len(imagedim) != 2:
            raise ValueError('The image dimensionality is not 2D.')
        if len(cases) != len(labels):
            raise ValueError('The length of labels must match the cases.')
        if len(segmentations) != len(labels):
            raise ValueError('The length of labels must match the segmentations.')

        self.projectFolder = projectfolder
        self.cases         = cases
        self.files         = files
        self.segmentations = segmentations
        self.segMask       = segmask
        self.y             = labels
        self.imageDim      = imagedim
        self.slices        = slices
        self.experiment    = experiment
        self.quiet         = quiet

        self.setupLogging(self.projectFolder)
        transcript.start(self.logFilename)

        if not self.quiet:
            print('Deep 2D Convolutional and Dense Autoencoder\n')
            print(f'experiment number: {self.testNumber}')
            print(f'log file name: {self.logFilename}')
            print(f'2D image/slice dimensions (y, x): {self.imageDim}')
        #endif

        self.getInputParameters()
        self.getImageData()
        self.getFolds()

        tic = time.perf_counter()

        self.makeEncoder()
        self.makeDecoder()
        self.makeCompileAutoEncoder()

        toc = time.perf_counter()
        print(f"Built and compiled the model in {toc - tic:0.1f} seconds")

        transcript.stop()

        self.testAutoEncoder()

        self.showPredictions()

    #enddef

    def showPredictions(self):
        import SimpleITK                  as     sitk
        import matplotlib.pyplot          as     plt

        # get_ipython().run_line_magic('matplotlib', 'inline')
        n = len(self.x_test)
        print(f'{n} slices to show.')
        for i in range(0, n):
            # Display original
            plt.figure(figsize=(1, 2))
            ax = plt.subplot(1, 2, 1)
            plt.imshow(self.x_test[i].reshape(self.imageDim))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax    = plt.subplot(1, 2, 2)
            arr   = self.decoded_imgs[i].reshape(self.imageDim)
            img   = sitk.GetImageFromArray(arr)
            case  = self.cases[self.test_cases[i]]
            sliceInCase = i % self.slices
            outfn = self.projectFolder + f'predicted_test{self.testNumber}_case{case}_slice{sliceInCase}.nii'
            sitk.WriteImage(img, outfn)
            plt.imshow(arr)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
        print('end')
    #enddef

    def testAutoEncoder(self):
        import time
        import numpy  as np
        import pandas as pd
        from   tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LambdaCallback

        tic     = time.perf_counter()
        if self.slices == 1:
            self.train_indices = list(self.train_index[0])
            self.train_cases   = list(self.train_index[0])
            self.test_indices  = list(self.test_index[0])
            self.test_cases    = list(self.test_index[0])
        else:
            self.train_indices = []
            self.train_cases   = []
            for i in list(self.train_index[0]):
                a =  i      * self.slices
                b = (i + 1) * self.slices
                c = [i]     * self.slices  # a list of the case index, repeated for each slice
                self.train_indices = self.train_indices + list(range(a, b))
                self.train_cases   = self.train_cases   + c
            # endfor
            self.test_indices = []
            self.test_cases   = []
            for i in list(self.test_index[0]):
                a =  i      * self.slices
                b = (i + 1) * self.slices
                c = [i]     * self.slices  # a list of the case index, repeated for each slice
                self.test_indices = self.test_indices + list(range(a, b))
                self.test_cases   = self.test_cases   + c
            # endfor
        # endif
        self.x_train = self.x[self.train_indices, :, :]  # add the need for list, to my python pet peeves
        self.x_test  = self.x[self.test_indices , :, :]

        # prevent overfitting with EarlyStopping (es) at minimum validation loss,
        # allowing for patience=15 epochs of no better value
        es      = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience)
        resting = LambdaCallback(on_epoch_begin=lambda epoch,logs: time.sleep(30) if ((epoch+1) % self.restEpoch) == 0 else 0)

        # run the autoencoder with a callback to es, and capturing history in loss_data
        loss_data = self.autoencoder.fit(self.x_train, self.x_train, epochs=self.maxEpochs,
                                         batch_size=self.batchSize, shuffle=True,
                                         validation_data=(self.x_test, self.x_test),
                                         callbacks=[es, resting, TensorBoard(log_dir='/tmp/autoencoder')])

        toc       = time.perf_counter()
        print(f"Trained and validated autoencoder in {toc - tic:0.1f} seconds")
        tic = toc

        loss_metric_df = pd.DataFrame(loss_data.history)
        hist_csv_file  = self.projectFolder + 'loss_measure_history_' + f'{self.testNumber}' + '.csv'
        with open(hist_csv_file, mode='w') as f:
            loss_metric_df.to_csv(f)
        toc = time.perf_counter()
        print(f"Wrote performance measure data to file in {toc - tic:0.1f} seconds")
        tic = toc

        toc = time.perf_counter()
        print(f"Saved the model in {toc - tic:0.1f} seconds")
        tic = toc

        self.decoded_imgs = self.autoencoder.predict(self.x_test)

        toc = time.perf_counter()
        print(f"Predicted with the model in {toc - tic:0.1f} seconds")

    #enddef

    def makeCompileAutoEncoder(self):
        from   tensorflow.keras           import Model

        self.autoencoder = Model(self.inputImage, self.decoded)
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.mainMeasure, metrics=self.otherMeasures)
    #enddef

    def makeDecoder(self):
        from   tensorflow.keras           import Model
        from   tensorflow.keras.layers    import Conv2DTranspose, UpSampling2D, Dense, Reshape

        u = 'UpSampl '
        t = 'Conv3DT '
        d = 'Dense   '
        r = 'Reshape '

        print('Decoder:')
        filters16    = 16
        filters8     = 8
        filters4     = 4
        kernelSizeC  = (3, 3)
        kernelSizeP  = (2, 2)
        kernelSizePP = (4, 4)

        x = Reshape(target_shape=self.convShape[1:])(self.encoded);                                   print(r, x.shape)
        x = UpSampling2D(              kernelSizeP)(x);                                               print(u, x.shape)
        x = Conv2DTranspose(filters16, kernelSizeC, padding='same', activation='relu', strides=1)(x); print(t, x.shape)
        x = UpSampling2D(              kernelSizeP)(x);                                               print(u, x.shape)
        x = Conv2DTranspose(filters16, kernelSizeC, padding='same', activation='relu', strides=1)(x); print(t, x.shape)
        x = UpSampling2D(              kernelSizeP)(x);                                               print(u, x.shape)
        x = Conv2DTranspose(filters16, kernelSizeC, padding='same', activation='relu', strides=1)(x); print(t, x.shape)
        x = UpSampling2D(              kernelSizeP)(x);                                               print(u, x.shape)
        x = Conv2DTranspose(filters16, kernelSizeC, padding='same', activation='relu', strides=1)(x); print(t, x.shape)
        #x = UpSampling2D(             kernelSizeP)(x);                                               print(u, x.shape)
        self.decoded = \
            Conv2DTranspose(1, kernelSizeC, padding='same', activation='relu', strides=2)(x); print(t, self.decoded.shape)
        self.decoder = Model(self.encoded, self.decoded)
    #enddef

    def makeEncoder(self):
        import numpy as np
        from   tensorflow.keras           import Input,  Model
        from   tensorflow.keras.layers    import Conv2D, MaxPooling2D, Dense, Reshape

        p = 'Padding '
        c = 'Conv3D  '
        d = 'Dense   '
        m = 'MaxPool '
        r = 'Reshape '

        print('Encoder:')
        channels     = 1
        self.imageShape   = self.imageDim + (channels,)  # (y, x, 1)
        self.inputImage   = Input(shape=self.imageShape); print(self.inputImage.shape)

        filters16    = 16
        filters8     = 8
        filters4     = 4
        kernelSizeC  = (3, 3)
        kernelSizeP  = (2, 2)
        kernelSizePP = (4, 4)

        # 5 layers...
        x = Conv2D(filters16, kernelSizeC,  padding='same', activation='relu')(self.inputImage); print(c, x.shape)
        x = MaxPooling2D(     kernelSizeP,  padding='same')(x);                                  print(m, x.shape)
        x = Conv2D(filters16, kernelSizeC,  padding='same', activation='relu', strides=1)(x);    print(c, x.shape)
        x = MaxPooling2D(     kernelSizeP,  padding='same')(x);                                  print(m, x.shape)
        x = Conv2D(filters16, kernelSizeC,  padding='same', activation='relu', strides=1)(x);    print(c, x.shape)
        x = MaxPooling2D(     kernelSizeP, padding='same')(x);                                   print(m, x.shape)
        x = Conv2D(filters16, kernelSizeC,  padding='same', activation='relu', strides=1)(x);    print(c, x.shape)
        x = MaxPooling2D(     kernelSizeP, padding='same')(x);                                   print(m, x.shape)
        x = Conv2D(filters16, kernelSizeC,  padding='same', activation='relu', strides=1)(x);    print(c, x.shape)
        x = MaxPooling2D(     kernelSizeP, padding='same')(x);                                   print(m, x.shape)
        self.convShape = x.shape
        self.dim = np.prod(self.convShape[1:])
        if self.dim > 25000:
            raise ValueError('The dimensions to Reshape before Dense layers, are too large')
        self.encoded = Reshape(target_shape=(-1, self.dim))(x);                            print(r, self.encoded.shape)
        self.encoder = Model(self.inputImage, self.encoded)
    #enddef

    def getFolds(self):
        from splitDataNd import getTrainAndValidationFoldIndices as getFolds
        from pandas      import Series
        y_s = Series(self.y)
        self.train_index, self.test_index = getFolds(y_s,  self.experiment['splitterType'],
                                                     self.experiment['folds'],
                                                     self.experiment['repeats'])
    #enddef

    def getImageData(self):
        import numpy     as     np
        import SimpleITK as     sitk
        import time

        tic = time.perf_counter()

        numCases         = len(self.cases)
        self.matrixDim   = (int(numCases * self.slices),) + self.imageDim  # store as 3D (even though we will process as 2D)
        self.x           = np.zeros(self.matrixDim)

        i = 0
        loadFailed = False
        for case, filename, segmentation in zip(self.cases, self.files, self.segmentations):
            print(f'Loading file: {filename}')
            img_dat = sitk.ReadImage(filename)
            img_mat = sitk.GetArrayFromImage(img_dat)
            loadedSlices    = img_mat.shape[0]
            loadedImageDimY = img_mat.shape[1]
            loadedImageDimX = img_mat.shape[2]
            if  self.slices      != loadedSlices    or \
                self.imageDim[0] != loadedImageDimY or \
                self.imageDim[1] != loadedImageDimX:
                print(f'Image dimensions {img_mat.shape} do not match expectations.')
                loadFailed = True
            else:
                # for a 2D image we get shape (1, 768, 1024)
                # for a 3D image we get shape (56, 512, 512)
                # same applies to the segmentation of course
                if self.useMask:
                   seg_dat = sitk.ReadImage(segmentation)
                   seg_mat = sitk.GetArrayFromImage(seg_dat)
                   mask    = self.segMask(seg_mat)
                   img_mat = np.multiply(img_mat, mask)
                # endif

                if self.slices == 1:
                    self.x[i, :, :] = img_mat[0, :, :]
                else:
                    a =  i    * self.slices
                    b = (i+1) * self.slices
                    self.x[a:b, :, :] = img_mat
                # endif
            # endif
            i += 1
        #endfor
        if loadFailed:
            raise ValueError('Load failed because image dimensions did not match expectations.')

        toc = time.perf_counter()
        if not self.quiet:
            print(f"Loaded data in {toc - tic:0.0f} seconds.")
    #enddef

    def getInputParameters(self):
        # the extra print commands are for the log file
        self.useMask       = input('Use segmentation mask (y/n): ')
        print(self.useMask)
        if self.useMask == 'y' or self.useMask == 'Y':
            self.useMask = True
        else:
            self.useMask = False
        self.optimizer     = input('Optimizer (e.g., Nadam, Adam, Adamax, Adadelta, Adagrad, RMSprop, SGD): ')
        print(self.optimizer)
        self.mainMeasure   = input('Measure to minimize (e.g., MSE, MAE): ')
        print(self.mainMeasure)
        maxEpochsText      = input('Maximum epochs (e.g., 400): ')
        self.maxEpochs     = int(maxEpochsText)
        print(maxEpochsText)
        patienceText       = input('Number of epochs allowed [patience] without improvement (e.g., 15): ')
        self.patience      = int(patienceText)
        print(patienceText)
        batchSizeText      = input('Batch size in slices (e.g., 8): ')
        self.batchSize     = int(batchSizeText)
        print(batchSizeText)
        encodingSizeText   = input('Requested image [bottleneck] encoding size (e.g., 100; or 300): ')
        self.encodingSize  = int(encodingSizeText)
        print(encodingSizeText)
        self.laptop        = input('Laptop: ')
        print(self.laptop)
        if self.laptop == 'y' or self.laptop == 'Y':
            self.laptop    = True
            restEpochText  = input('Rest for 30s every x epochs: ')
            self.restEpoch = int(restEpochText)
        else:
            self.laptop    = True
            self.restEpoch = 0
        #endif

        # aside from the mainMeasure, get a list of the other measures to also log
        self.otherMeasures = []
        for measure in ['MSE', 'MAE', 'binary_crossentropy', 'cosine_similarity']:
            if str.lower(measure) == str.lower(self.mainMeasure):
                continue
            self.otherMeasures = self.otherMeasures + [measure]
        # endfor
        print(f'Other measures:      {self.otherMeasures}')

        return
    #enddef

    def setupLogging(self, project_folder):
        import sys
        import warnings
        import os
        import acLogging as log

        # turn off system warnings
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
        # endif

        # get next number for logfile and begin logging
        self.project_folder = project_folder
        if self.project_folder[-1] == '/':
            self.project_folder = self.project_folder[0:-1]
        filenamePrefix = f'{self.project_folder}/log_CAE2D_'
        filenameSuffix = '.txt'
        self.logFilename, self.testNumber = log.findNextFileNumber(filenamePrefix, filenameSuffix)
        return
    #enddef

#endclass
