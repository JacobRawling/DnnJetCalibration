import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc


def load_data(test_csv, train_csv):
    return pd.read_csv(test_csv), pd.read_csv(train_csv)


df_test, df_train = load_data('/hepgpu3-data1/jrawling/deep_jets/csvs/tag_1_delphes_jets_test.csv',
                              '/hepgpu3-data1/jrawling/deep_jets/csvs/tag_1_delphes_jets_train.csv')

print('Loaded test and train data.')
class PandasPlot:
    def __init__(self, variable, color, label):
        self.variable=variable
        self.label=label
        self.color=color

def create_plots(
     df,
     variables,
     binning=None,
     xlabel='',
     ylabel='',
     filename='plot.png',
     ) :
    plt.figure()
    for var in variables:
        plt.hist(df[var.variable].values, fc=(
        1, 0, 0, 0),  edgecolor=var.color, label=var.label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)

eta_plots=[ PandasPlot('calib_jet_eta_0', 'green', 'MLP'),
            PandasPlot('truth_jet_eta_0', 'orange', 'truth'),
            PandasPlot('jet_eta_0', 'red', 'Uncalib'),
            ]
phi_plots=[ PandasPlot('calib_jet_phi_0', 'green', 'MLP'),
            PandasPlot('truth_jet_phi_0', 'orange', 'truth'),
            PandasPlot('jet_phi_0', 'red', 'Uncalib'),
            ]
pt_plots=[ PandasPlot('calib_jet_pt_0', 'green', 'MLP'),
            PandasPlot('truth_jet_pt_0', 'orange', 'truth'),
            PandasPlot('jet_pt_0', 'red', 'Uncalib'),
            ]
mass_plots=[ PandasPlot('calib_jet_m_0', 'green', 'MLP'),
            PandasPlot('truth_jet_m_0', 'orange', 'truth'),
            PandasPlot('jet_m_0', 'red', 'Uncalib'),
            ]

create_plots(
    df=df_train,
    variables=phi_plots,
    xlabel=r'Jet \phi',
    ylabel=r'Number of events',
    filename='out/phi.png'
    )

create_plots(
    df=df_train,
    variables=eta_plots,
    xlabel=r'Jet \eta',
    ylabel=r'Number of events',
    filename='out/eta.png'
    )
create_plots(
    df=df_train,
    variables=pt_plots,
    xlabel=r'Jet $p_{t} [$MeV$]$',
    ylabel=r'Number of events',
    filename='out/pt.png'
    )
create_plots(
    df=df_train,
    variables=mass_plots,
    xlabel=r'Jet mass [$MeV$]$',
    ylabel=r'Number of events',
    filename='out/mass.png'
    )
