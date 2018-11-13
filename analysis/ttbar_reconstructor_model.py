from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

from keras.layers import ELU, Input
import keras.layers as kel
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras import regularizers
import glob 
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import matplotlib

matplotlib.use('agg')
import os
import matplotlib.pyplot as plt 
plt.style.use('classic')
import seaborn as sn

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" - val_f1: %f - val_recall: %f"%(_val_f1, _val_recall))
        return
 
class TTbarReconstructor:
    def __init__(self, 
                 input_file_path,
                 input_columns,
                 categories=[0,1,2,3,4,5],
                 layers=[10],
                 epochs=1,
                 ):
        self.in_files = glob.glob(input_file_path)
        self.categories = categories
        self.input_columns = input_columns
        self.training_iteration = 0 
        self.output_folder = 'out/'
        self.epochs=epochs
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.df = \
            self.load_train_test_data()

        self.plot_correlation()

    def load_train_test_data(self):
        df = pd.read_csv(self.in_files[0])
        for f in self.in_files[1:]:
            df = pd.concat([df, pd.read_csv(f)],sort=False)

        mask = (df['truth_p%i'%self.categories[0]]==1) 
        for c in self.categories[1:]:
            mask = mask | (df['truth_p%i'%c]==1) 
        df = df[mask]
        df = df[df['n_bjets'] == 2]

        n_events = len(df)
        self.class_weight = {}
        for i,c in enumerate(self.categories):
            self.class_weight[i] = float(n_events)/float(len(df[df['truth_p%i'%c]==1]))

        print('Class Weights: ', self.class_weight)
        
        df['furthest_from_W']  =np.argmax(df[['dR_Wj%d'%i for i in range(5)]].values,axis=1)
        df['closest_to_met']  =np.argmin(df[['j%d_met_dPhi'%i for i in range(5)]].values,axis=1)
        df['furthest_from_met']  =np.argmax(df[['j%d_met_dPhi'%i for i in range(5)]].values,axis=1)
        df['closest_to_lep']  =np.argmin(df[['dR_Lj%d'%i for i in range(5)]].values,axis=1)
        df['furthest_from_lep']  =np.argmax(df[['dR_Lj%d'%i for i in range(5)]].values,axis=1)

        print(df.head())

        # # Create the test and train data
        X = df[self.input_columns]

        # The truth information 
        truth_columns = [ 'truth_p%d'%i for i in self.categories] 
        Y = df[truth_columns] 

        print('Input Variables:' )
        for t in self.input_columns:
            print('\t',t)


        # return X,Y
        # # Split the data into test and train samples
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=0)
        return X_train, X_test, Y_train, Y_test, df


    def mlp_model(self, input_d, output_d, layers=[10]):
        previous_layer = input_layer = Input(shape=(input_d, ))

        for i, l in enumerate(layers):
            previous_layer = Dense(l, activation='elu',
                            kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01))(previous_layer)
            if i != 0:
                # previous_layer = Dropout(0.05)(previous_layer)
                previous_layer = BatchNormalization()(previous_layer)

        final_layer = Dense(output_d, activation='softmax',)(previous_layer)
        model = Model(inputs=[input_layer], outputs=final_layer)
        return model 

    def train_model(self):
        X_train, X_test, Y_train, Y_test = self.X_train, self.X_test, self.Y_train, self.Y_test

        self.model = self.mlp_model(input_d=len(X_train.values[0]), 
                                    output_d=len(self.categories),
                                    layers=self.layers)
        print(self.model.summary())

         # Compile model
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc' ],
                      )
        plot_model(self.model, to_file=self.output_folder + 'model.eps',show_shapes=True)

        self.metrics = Metrics()
        self.history = self.model.fit(X_train,Y_train,
              validation_split=0.3,
              epochs=self.epochs,
              callbacks=[self.metrics,EarlyStopping(patience=2),   
                         ModelCheckpoint(filepath=self.output_folder+"weights.best.hdf5", save_best_only=True)],
              class_weight=self.class_weight,)


    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.output_folder+'model.json', "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(self.output_folder+"model.h5")
        print("Saved model to disk: out/model.h5")

    def plot_f1_score(self):
        plt.figure(figsize = (10,7))
        f1_scores = self.metrics.val_f1s
        plt.plot(f1_scores)
        plt.xlabel('Epocs')
        plt.ylabel('F1 Score') 
        plt.savefig(self.output_folder+'F1_scrore.png')
        plt.show()

    def plot_accuracy_and_loss(self):
        # summarize history for accuracy
        fig, ax1 = plt.subplots()
        # plt.plot(history.history['acc'])
        l1 = ax1.plot(self.history.history['val_acc'], color='g',label='validation accuracy')
        # plt.title('model accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.0)
        ax1.yaxis.label.set_color('g')

        # draw_ATLAS_labels(plt.gca())
        # plt.savefig(self.output_folder + 'accuracy.png')

        # summarize history for loss
        ax2 = ax1.twinx()
        l2 = ax2.plot(self.history.history['loss'],color='cornflowerblue', label='Train loss')
        l3 = ax2.plot(self.history.history['val_loss'],color='tomato', label='Test loss')
        ax2.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        plt.xlabel('Epoch')
        lns = l1+l2+l3
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, edgecolor=None,facecolor=None,loc='center right')
        # draw_ATLAS_labels(plt.gca())
        plt.savefig(self.output_folder + 'acc_loss.png')
        plt.show()

    def plot_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        Y_pred = self.model.predict(self.X_test)
        Y_pred = np.argmax(Y_pred, axis=1) 
        Y_test = np.argmax(self.Y_test.values, axis=1) 

        print('Y_pred', Y_pred)
        print('Y_test', Y_test)

        print('Y_pred:', set(Y_pred))
        print('Y_test:', set(Y_test))
        cm = confusion_matrix(Y_test, Y_pred)

        df_cm = pd.DataFrame(cm)#, index=[str(i) for i in xrange(n_categories)],
                      # columns=[str(i) for i in xrange(n_categories)])

        confusion_matrix = 100.0*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(confusion_matrix)
        print("Normalized confusion matrix")
        plt.figure(figsize = (10,7))
        sn.heatmap(confusion_matrix, annot=True,cmap="Greens")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.output_folder + 'confusion_matrix.png')
        plt.show()
        print('Saved confusion_matirx.png')

    def plot_correlation(self):
        sn.set(style="white")
        plt.rc('text', usetex=False)

        # Compute the correlation matrix
        cols = self.input_columns 
        cols += ['truth_param'] 
        corr = self.df[cols].corr()

        # # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 11))

        # Generate a custom diverging colormap
        cmap = sn.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sn.heatmap(corr,cmap=cmap,mask=mask, center=0,
                    square=False, linewidths=.5, cbar_kws={"shrink": .5})

        plt.savefig('Correlation.png') 

    def __call__(self, layers):
        print('Creating new model with layers:', layers)
        self.layers = layers
        self.training_iteration += 1
        self.output_folder = 'out/iteration_%d/'%(self.training_iteration)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.train_model()
        self.save_model()
        self.plot_accuracy_and_loss()
        self.plot_confusion_matrix()
        self.plot_f1_score()
        f1_score = np.max(self.metrics.val_f1s)
        print('For configuration:', layers)
        print('F1 Scores: ', self.metrics.val_f1s)
        print('Final F1 Score: ',  f1_score)

        K.clear_session()
        del self.model
        return  f1_score



def main():
    train_columns = [ 'best_had_m_perm',
           'best_lep_m_perm',
           'W_lep_eta',
           'closest_to_W',
           'n_bjets',
           'furthest_from_W',
           'closest_to_met',
           'furthest_from_met',
           'closest_to_lep',
           'furthest_from_lep',
           'met_phi',
           'lep_pt',
           'best_reco_param'
           ]

    train_columns = train_columns + ['dR_Lj%d'%i for i in range(5)]
    train_columns = train_columns + ['dR_Wj%d'%i for i in range(5)]
    train_columns = train_columns + ['jet_isBjet_%d'%i for i in range(5)]

    ttbar_reconstructor = TTbarReconstructor(
        input_file_path='/hepgpu3-data1/jrawling/deep_tops/csvs_test/ttbar_0.csv',
        input_columns=train_columns,
        )
    ttbar_reconstructor(layers=[10,10])
main()
    