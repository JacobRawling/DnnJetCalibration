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
from keras.callbacks import EarlyStopping
from keras import regularizers
import glob 
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt 
plt.style.use('classic')
import seaborn as sn

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif',size=50)
categories = [0,1,2,3,4,5]
n_categories = len(categories)
class_weight = {}

def load_train_test_data():
    in_files = glob.glob('/hepgpu3-data1/jrawling/deep_tops/csvs_test/ttbar_0.csv')
    df = pd.read_csv(in_files[0])
    for f in in_files[1:]:
        df = pd.concat([df, pd.read_csv(f)],sort=False)

    mask = (df['truth_p%i'%categories[0]]==1) 
    for c in categories[1:]:
        mask = mask | (df['truth_p%i'%c]==1) 
    df = df[mask]
    df = df[df['n_bjets'] == 2]

    n_events = len(df)
    for i,c in enumerate(categories):
        class_weight[i] = float(n_events)/float(len(df[df['truth_p%i'%c]==1]))
        print('Cat ', c, ' weight = ',len(df[df['truth_p%i'%c]==1]), '/', n_events, ' = ',class_weight[i]  )
    print('Class Weights: ', class_weight)
    
    df['furthest_from_W']  =np.argmax(df[['dR_Wj%d'%i for i in range(5)]].values,axis=1)
    df['closest_to_met']  =np.argmin(df[['j%d_met_dPhi'%i for i in range(5)]].values,axis=1)
    df['furthest_from_met']  =np.argmax(df[['j%d_met_dPhi'%i for i in range(5)]].values,axis=1)
    df['closest_to_lep']  =np.argmin(df[['dR_Lj%d'%i for i in range(5)]].values,axis=1)
    df['furthest_from_lep']  =np.argmax(df[['dR_Lj%d'%i for i in range(5)]].values,axis=1)

    print(df.head())
    # # Create the test and train data
    train_columns = [ 
           'best_had_m_perm',
           'best_lep_m_perm',
           # 't_had_m_p_0', 't_had_m_p_1', 't_had_m_p_2', 't_had_m_p_3', 't_had_m_p_4', 't_had_m_p_5',
           # 't_lep_m_p_0', 't_lep_m_p_1', 't_lep_m_p_2', 't_lep_m_p_3', 't_lep_m_p_4', 't_lep_m_p_5',
           u'W_lep_eta',
           # u'W_lep_m',
           # u'W_lep_phi', 
           # u'W_lep_pt', 
           # u'jet_e_0', u'jet_e_1', u'jet_e_2', u'jet_e_3', u'jet_e_4',
           # u'jet_eta_0', u'jet_eta_1', u'jet_eta_2', u'jet_eta_3', u'jet_eta_4',
           # u'jet_phi_0', u'jet_phi_1', u'jet_phi_2', u'jet_phi_3',
           # u'jet_phi_4', 
           # u'jet_pt_0', u'jet_pt_1', u'jet_pt_2', u'jet_pt_3', u'jet_pt_4', 
           # u'met'
           'closest_to_W',
           'n_bjets',
           'furthest_from_W',
           'closest_to_met',
           'furthest_from_met',
           'closest_to_lep',
           'furthest_from_lep',
           # 'met_met',
           'met_phi',
           'lep_pt',
           # 'lep_phi'
           # 'truth_param'
           'best_reco_param'
           ]

    train_columns = train_columns + ['dR_Lj%d'%i for i in range(5)]
    train_columns = train_columns + ['dR_Wj%d'%i for i in range(5)]
    train_columns = train_columns + ['jet_isBjet_%d'%i for i in range(5)]
    X = df[train_columns]

    # The truth information 
    truth_columns = [ 'truth_p%d'%i for i in categories] 
    Y = df[truth_columns] 

    print('Input Variables:' )
    for t in train_columns:
        print('\t',t)
    # return X,Y
    # # Split the data into test and train samples
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    return X_train, X_test, Y_train, Y_test

def draw_ATLAS_labels(ax, messages=[]):
    """
    Draw the ATLAS label for a Matplotlib plot .
    """
    plt.rc('font', family='sans-serif',size=25)
    plt.text(0.05, 0.95, '\\textbf{ATLAS} Internal', fontsize=30, transform=ax.transAxes)
    plt.text(0.05, 0.93, '$\\sqrt{s}=13$ TeV', fontsize=22, transform=ax.transAxes)
    y = 0.91
    for msg in messages:
        plt.text(0.05, y, msg, fontsize=22, transform=ax.transAxes)
        y -= 0.02
    plt.rc('font', family='serif',size=25)

def mlp_model(input_d, output_d, layers=[10]):
    previous_layer = input_layer = Input(shape=(input_d, ))

    for i, l in enumerate(layers):
        previous_layer = Dense(l, activation='elu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(previous_layer)
        if i != 0:
            previous_layer = Dropout(0.05)(previous_layer)
            previous_layer = BatchNormalization()(previous_layer)

    final_layer = Dense(output_d, activation='softmax',)(previous_layer)
    model = Model(inputs=[input_layer], outputs=final_layer)



def feedforward_model(input_d=4, spectator_d=10):
    """
    Creates a simple feed power NN that takes a four vector as an input and 
    returns a calibrated 4 vector 
    """
    

    first_input = Input(shape=(input_d, ))
    first_dense = Dense(30, activation='elu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(first_input)
    # first_dropout = Dropout(0.3)(first_dense)
    # first_batchnorm = BatchNormalization()(first_dropout)

    second_dense = Dense(150, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)) (first_dense)
    second_dropout = Dropout(0.05)(second_dense)
    second_batchnorm = BatchNormalization()(second_dense)

    third_dense = Dense(50, activation='elu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(second_batchnorm)
    # third_dropout = Dropout(0.1)(third_dense)

    fourth_dense = Dense(30, activation='elu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01))(third_dense)
    fourth_batchnorm = BatchNormalization()(fourth_dense)

    final_layer = Dense(n_categories, activation='softmax',)(fourth_batchnorm)
    model = Model(inputs=[first_input], outputs=final_layer)

    return model 

def train_model():
    X_train, X_test, Y_train, Y_test = load_train_test_data()

    model = feedforward_model(input_d=len(X_train.values[0]), 
                              spectator_d=None)
    print(model.summary())

     # Compile model
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc','top_k_categorical_accuracy','categorical_accuracy' ],
                  )
    plot_model(model,to_file='out/model.eps',show_shapes=True)

    # Fit Model
    history = model.fit(X_train,Y_train,
              validation_split=0.3,
              epochs=20,
              # callbacks=[EarlyStopping(patience=4)],
              class_weight=class_weight,)
   
    # serialize model to JSON
    model_json = model.to_json()
    with open("out/model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("out/model.h5")
    print("Saved model to disk: out/model.h5")

    print(history.history.keys())

    print('acc: ', history.history['acc'])
    print('val_acc: ', history.history['val_acc'])
    print('val_loss: ', history.history['val_loss'])
    # summarize history for accuracy
    fig, ax1 = plt.subplots()
    # plt.plot(history.history['acc'])
    l1 = ax1.plot(history.history['val_acc'], color='g',label='validation accuracy')
    # plt.title('model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.0)
    # draw_ATLAS_labels(plt.gca())
    # plt.savefig('out/accuracy.png')

    # summarize history for loss
    ax2 = ax1.twinx()
    l2 = ax2.plot(history.history['loss'],color='cornflowerblue', label='Train loss')
    l3 = ax2.plot(history.history['val_loss'],color='tomato', label='Test loss')
    ax2.set_ylabel('Loss')

    plt.xlabel('Epoch')
    lns = l1+l2+l3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, edgecolor=None,facecolor=None,loc='center right')
    # draw_ATLAS_labels(plt.gca())
    plt.savefig('out/acc_loss.png')


    from sklearn.metrics import confusion_matrix
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1) 
    Y_test = np.argmax(Y_test.values, axis=1) 

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
    plt.savefig('out/confusion_matrix.png')
    print('Saved confusion_matirx.png')
train_model()
