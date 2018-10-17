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
        input_file='/hepgpu3-data1/jrawling/deadcone_AT_tuples/4104070_mc16a/user.jrawling.15477193._000668.output.root',
        output_folder='/hepgpu3-data1/jrawling/deep_tops/csvs/',
        output_file_name='ttbar.csv',
        variables_to_save={
            # Reco level
            'n_jets': 'Length$(jet_pt)',
            # 'jet_pt_0': 'Jet.PT[0]',
            # 'jet_eta_0': 'Jet.Eta[0]',
            # 'jet_phi_0': 'Jet.Phi[0]',
            # 'jet_m_0': 'Jet.Mass[0]',

            # Ground truth
            # 'truth_had_top_pt': 'GenJet.PT[0]',
            'truth_had_top_eta': 'truth.MC_tbar_afterFSR_eta',
            # 'truth_had_top_phi': 'GenJet.Phi[0]',
            # 'truth_had_top_m': 'GenJet.Mass[0]',
            # 'truth_n_jets': 'GenJet_size'
        },
        selection='',
        tuple_name='particleLevel',
        friend_tuple_name='truth',
        major_index="(runNumber<<13)",
        minor_index="eventNumber"

    )


main()
