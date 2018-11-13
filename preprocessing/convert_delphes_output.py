"""

"""

import mljets as mj
import ROOT as r
from tqdm import tqdm


def main():
    # Construct the tchain of the input files
    mj.load_delphes_library()

    # Run the conversion
    mj.root_to_csv(
        input_file='/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/dijet_2/Events/run_03/tag_1_delphes_events.root',
        output_folder='/hepgpu3-data1/jrawling/deep_jets/csvs/',
        output_file_name='tag_1_delphes_jets.csv',
        variables_to_save={
            # Reco level
            'jet_pt_0': 'Jet.PT[0]',
            'jet_eta_0': 'Jet.Eta[0]',
            'jet_phi_0': 'Jet.Phi[0]',
            'jet_m_0': 'Jet.Mass[0]',
            'n_jets': 'Jet_size',

            # Ground truth
            'truth_jet_pt_0': 'GenJet.PT[0]',
            'truth_jet_eta_0': 'GenJet.Eta[0]',
            'truth_jet_phi_0': 'GenJet.Phi[0]',
            'truth_jet_m_0': 'GenJet.Mass[0]',
            'truth_n_jets': 'GenJet_size'
        },
        selection='(Jet_size>=1)*( ((Jet.Eta[0]-GenJet.Eta[0])**2 +(Jet.Phi[0]-GenJet.Phi[0])**2)**0.5 < 0.4 )   ',
        tuple_name='Delphes'
    )


main()
