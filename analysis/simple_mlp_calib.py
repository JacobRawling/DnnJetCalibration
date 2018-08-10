from __future__ import print_function
from mljets.models import feedforward_model
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

def main():
    # Grab the model from our avaliable set of models
    model = feedforward_model()

    # Compile model
    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['accuracy'])

    # Load the data
    in_file = '/hepgpu3-data1/jrawling/deep_jets/csvs/tag_1_delphes_jets.csv'
    df = pd.read_csv(in_file)

    # Create the test and train data
    truth_columns = ['truth_jet_eta_0', 'truth_jet_m_0',
                     'truth_jet_phi_0', 'truth_jet_pt_0']
    train_columns = ['jet_eta_0', 'jet_m_0', 'jet_phi_0', 'jet_pt_0']
    Y = df[truth_columns]
    X = df[train_columns]

    # Split the data into test and train samples
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)

    # Fit Model
    model.fit(Y_train, Y_train)
    # model.fit(X_train, Y_train)

    # evaluate the model
    scores = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Save the processed variable
    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)

    # 
    out_test = np.concatenate( (X_test.values, Y_test.values, Y_test_pred), axis=1)
    df_test = pd.DataFrame(out_test,
                           columns=train_columns + truth_columns +
                           [c.replace('truth', 'calib') for c in truth_columns]
                           )

    out_train = np.concatenate( (X_train.values, Y_train.values, Y_train_pred), axis=1)
    df_train = pd.DataFrame(out_train,
                           columns=train_columns + truth_columns +
                           [c.replace('truth', 'calib') for c in truth_columns]
                           )
    # Now save this 
    df_train.to_csv(in_file.replace('.csv','_train.csv'))
    df_train.to_csv(in_file.replace('.csv','_test.csv'))
    
    print('Saving train data to:', in_file.replace('.csv','_train.csv'))
    print('Saving test data to:', in_file.replace('.csv','_test.csv'))

main()
