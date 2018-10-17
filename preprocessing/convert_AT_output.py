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
            'jet_pt_0': 'jet_pt[0]/1e3',
            'jet_eta_0': 'jet_eta[0]',
            'jet_phi_0': 'jet_phi[0]',
            'jet_e_0': 'jet_e[0]/1e3',

            'jet_isBjet_0': '(jet_nGhosts_bHadron[0] > 0)',
            'jet_pt_1': 'jet_pt[1]/1e3',
            'jet_eta_1': 'jet_eta[1]',
            'jet_phi_1': 'jet_phi[1]',
            'jet_e_1': 'jet_e[1]/1e3',

            'jet_isBjet_1': '(jet_nGhosts_bHadron[1] > 0)',
            'jet_pt_2': 'jet_pt[2]/1e3',
            'jet_eta_2': 'jet_eta[2]',
            'jet_phi_2': 'jet_phi[2]',
            'jet_e_2': 'jet_e[2]/1e3',
            'jet_isBjet_2': '(jet_nGhosts_bHadron[2] > 0)',

            'jet_pt_3': 'jet_pt[3]/1e3',
            'jet_eta_3': 'jet_eta[3]',
            'jet_phi_3': 'jet_phi[3]',
            'jet_e_3': 'jet_e[3]/1e3',
            'jet_isBjet_3': '(jet_nGhosts_bHadron[3] > 0)',

            'jet_pt_4': 'jet_pt[4]/1e3',
            'jet_eta_4': 'jet_eta[4]',
            'jet_phi_4': 'jet_phi[4]',
            'jet_e_4': 'jet_e[4]/1e3',
            'jet_isBjet_4': '(jet_nGhosts_bHadron[4] > 0)',

            'W_lep_pt': 'jet_pt[4]/1e3',
            'W_lep_eta': 'jet_eta[4]',
            'W_lep_phi': 'jet_phi[4]',
            'W_lep_m': 'jet_e[4]/1e3',

            'met': 'met_met',

            # Ground truth
            # 'truth_had_top_pt': 'GenJet.PT[0]',
            'truth_had_top_pt':  'truth.MC_tbar_afterFSR_pt/1e3',
            'truth_had_top_eta': 'truth.MC_tbar_afterFSR_eta',
            'truth_had_top_phi': 'truth.MC_tbar_afterFSR_phi',
            'truth_had_top_m':   'truth.MC_tbar_afterFSR_m/1e3',

            'truth_lep_top_pt':  'truth.MC_tbar_afterFSR_pt/1e3',
            'truth_lep_top_eta': 'truth.MC_tbar_afterFSR_eta',
            'truth_lep_top_phi': 'truth.MC_tbar_afterFSR_phi',
            'truth_lep_top_m':   'truth.MC_tbar_afterFSR_m/1e3',
            # 'truth_had_top_phi': 'GenJet.Phi[0]',
            # 'truth_had_top_m': 'GenJet.Mass[0]',
            # 'truth_n_jets': 'GenJet_size'
        },
        selection='1.0',
        tuple_name='particleLevel',
        friend_tuple_name='truth',
        major_index="(runNumber<<13)",
        minor_index="eventNumber"

    )


main()
