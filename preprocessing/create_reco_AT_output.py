"""

"""
from __future__ import print_function
import mljets as mj
import ROOT as r
from tqdm import tqdm
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import numpy as np
from keras.models import model_from_json


def convert_elements_to_list(dictionary):
    """
    @Brief
    Takes the elements of the inputted dictioanry and returns a new dictioanry
    with those elements in al list. 

    @Params
    dictionary: A Dictionary, OrderedDit, or any object that can iterate by
                a key. 
    """
    for name in dictionary:
        dictionary[name] = [dictionary[name]]
    return dictionary

def get_jets_from_indices(e, indicies):
    jets = []
    for b in indicies:
        jet = r.TLorentzVector()
        jet.SetPtEtaPhiM(
                e.jet_pt[b],
                e.jet_eta[b],
                e.jet_phi[b],
                e.jet_e[b],
            )
        jets.append(jet) 
    return jets 

def get_ttbar(jets, e):
    t_had = jets[1] + jets[-2] + jets[-1]
    t_lep = jets[0] + e.W_lep
    return t_had, t_lep 

def cost_function(t_had, t_lep, truth_t_had, truth_t_lep):
    mse = (t_had.Eta() - truth_t_had.Eta())**2 
    mse += (t_had.Phi() - truth_t_had.Phi())**2
    mse += (t_had.Pt() - truth_t_had.Pt())**2 
    mse += (t_had.M()/t_had.E() - truth_t_had.M()/truth_t_had.E() )**2 

    mse = (t_lep.Eta() - truth_t_lep.Eta())**2 
    mse += (t_lep.Phi() - truth_t_lep.Phi())**2
    mse += (t_lep.Pt() - truth_t_lep.Pt())**2 
    mse += (t_lep.M()/t_lep.E() - truth_t_lep.M()/truth_t_lep.E() )**2 

    # 8 DoF we're averaging over  
    mse = mse/4.0/4.0
    return mse

def get_permutation_vector(n):
    b_inital_permutation, l_inital_permutation = [0,1], [2,3,4]
    i = 0
    for b_i in multiset_permutations(b_inital_permutation):
        for l_i in multiset_permutations(l_inital_permutation):
            if i == n:
                return b_i + l_i
            i+=1 
    return None

def get_best_permutaiton(e, truth_t_had, truth_t_lep ):
    initial_permutation, cost = [0,1,2,3,4], {}
    b_inital_permutation, l_inital_permutation = [0,1], [2,3,4]
    i = 0
    for b_i in multiset_permutations(b_inital_permutation):
        for l_i in multiset_permutations(l_inital_permutation):
            # 
            jets = get_jets_from_indices(e, b_i+l_i)
            t_had, t_lep = get_ttbar(jets, e)

            # Reconstruct the top quark 
            L = cost_function( 
                            t_had, t_lep,
                            truth_t_had, truth_t_lep 
                        )
            cost[i] = L
            i += 1

    min_index =  min(cost, key=cost.get)
    return min_index, cost.keys()

def get_jets(e, b_indices, l_indices):
    jets = []
    for b in b_indices:
        jet = r.TLorentzVector()
        jet.SetPtEtaPhiM(
                e.jet_pt[b],
                e.jet_eta[b],
                e.jet_phi[b],
                e.jet_e[b],
            )
        jets.append(jet)
    for l in l_indices:
        jet = r.TLorentzVector()
        jet.SetPtEtaPhiM(
                e.jet_pt[l],
                e.jet_eta[l],
                e.jet_phi[l],
                e.jet_e[l],
            )
        jets.append(jet)
    return jets


def get_truth_ttbar(variables):
    # 
    t_had, t_lep = r.TLorentzVector(), r.TLorentzVector()
    
    t_had.SetPtEtaPhiM(
            variables['truth_had_top_pt'],
            variables['truth_had_top_eta'],
            variables['truth_had_top_phi'],
            variables['truth_had_top_m'],
        )
    t_lep.SetPtEtaPhiM(
            variables['truth_lep_top_pt'],
            variables['truth_lep_top_eta'],
            variables['truth_lep_top_phi'],
            variables['truth_lep_top_m'],
        )

    return t_had, t_lep 

def get_model_input(variables):

    train_columns = [ u'W_lep_eta', u'W_lep_m', u'W_lep_phi', u'W_lep_pt',
           u'jet_e_0', u'jet_e_1', u'jet_e_2', u'jet_e_3', u'jet_e_4',
           u'jet_eta_0', u'jet_eta_1', u'jet_eta_2', u'jet_eta_3', u'jet_eta_4',
           u'jet_isBjet_0', u'jet_isBjet_1', u'jet_isBjet_2', u'jet_isBjet_3',
           u'jet_isBjet_4', u'jet_phi_0', u'jet_phi_1', u'jet_phi_2', u'jet_phi_3',
           u'jet_phi_4', u'jet_pt_0', u'jet_pt_1', u'jet_pt_2', u'jet_pt_3',
           u'jet_pt_4', u'met', u'n_jets']
    df_temp = pd.DataFrame.from_dict(convert_elements_to_list(
                    variables
                ))
    X = df_temp[train_columns]
    return X 

def append_reco_variables(variables, t_had, t_lep):
    variables["NN_t_lep_pt"] = t_lep.Pt()
    variables["NN_t_lep_eta"] = t_lep.Eta()
    variables["NN_t_lep_phi"] = t_lep.Phi()
    variables["NN_t_lep_m"] = t_lep.M()

    variables["NN_t_had_pt"] = t_had.Pt()
    variables["NN_t_had_eta"] = t_had.Eta()
    variables["NN_t_had_phi"] = t_had.Phi()
    variables["NN_t_had_m"] = t_had.M()

    return variables

def append_resolution_variables(variables, t_had, t_lep,truth_t_had, truth_t_lep, e):

    variables["NN_t_lep_pt_R"]  = t_lep.Pt()/truth_t_lep.Pt()
    variables["NN_t_lep_eta_R"] = t_lep.Eta() - truth_t_lep.Eta()
    variables["NN_t_lep_phi_R"] = t_lep.Phi() - truth_t_lep.Phi()
    variables["NN_t_lep_m_R"]   = t_lep.M()/truth_t_lep.M()

    variables["NN_t_had_pt_R"]  = t_had.Pt()/truth_t_had.Pt()
    variables["NN_t_had_eta_R"] = t_had.Eta() - truth_t_had.Eta()
    variables["NN_t_had_phi_R"] = t_had.Phi() - truth_t_had.Phi()
    variables["NN_t_had_m_R"]   = t_had.M()/truth_t_had.M()
    
    t_lep, t_had = e.top_lep, e.top_had 
    variables["t_lep_pt_R"]  = t_lep.Pt()/truth_t_lep.Pt()
    variables["t_lep_eta_R"] = t_lep.Eta() - truth_t_lep.Eta()
    variables["t_lep_phi_R"] = t_lep.Phi() - truth_t_lep.Phi()
    variables["t_lep_m_R"]   = t_lep.M()/truth_t_lep.M()

    variables["t_had_pt_R"]  = t_had.Pt()/truth_t_had.Pt()
    variables["t_had_eta_R"] = t_had.Eta() - truth_t_had.Eta()
    variables["t_had_phi_R"] = t_had.Phi() - truth_t_had.Phi()
    variables["t_had_m_R"]   = t_had.M()/truth_t_had.M()
    return variables

def root_to_csv_jet_assignmet(
            input_file,
            output_folder,
            output_file_name,
            tuple_name,
            variables_to_save={},
            selection='1.0',
            friend_tuple_name=None,
            major_index=None,
            minor_index=None
            ):
    """
    @Brief
    Converts an arbitrary ROOT file to a CSV file. 

    @Parameters
    input_file: Full directory of the input_file
    ouput_folder: Full path of output folder
    output_file: Name of the output file. 
    variables_to_Save: A dictionary whose key is the name of a column in the csv
                       and element is a string to process on the TTree 
    selection: Event selection criteria to apply on the ttree. 
    """
    # Construct the tchain of the input files
    chain = r.TChain(tuple_name)
    if isinstance(input_file, str):
        chain.AddFile(input_file)
    else: 
        for f in input_file:    
            chain.AddFile(f)


    if friend_tuple_name is not None:
        friend_chain = r.TChain(friend_tuple_name)
        if isinstance(input_file, str):
            friend_chain.AddFile(input_file)
        else: 
            for f in input_file:    
                friend_chain.AddFile(f)
        # Add the frirends
        nominal_index = r.TTreeIndex(chain,major_index,minor_index)
        chain.SetTreeIndex(nominal_index)

        friend_index = r.TTreeIndex(friend_chain,major_index,minor_index)
        friend_chain.SetTreeIndex(friend_index)
        chain.AddFriend(friend_chain)


    # Create the TTreeForumlas based on the variables_to_save to allow us
    # to easily add inputs in a TTree::Draw style
    formulae, variables = {}, {}
    for name in variables_to_save:
        formulae[name] = r.TTreeFormula(name + "_formula",
                                        variables_to_save[name], chain)

    selection_formula = r.TTreeFormula(
        'selection_formula', selection, chain)
    # Set-up blank dataframe
    df = None

    def load_model(model_json_path, model_weight_path):
        # load json and create model
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weight_path)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return loaded_model

    model = load_model('out/model.json', 'out/model.h5')

    # # Iterate over the tree and
    with tqdm(total=chain.GetEntries()) as progress_bar:
        for e in chain:
            progress_bar.update(1)

            # Only evaluate variables that pass a set of selection criteria
            pass_selection = selection_formula.EvalInstance()
            if not pass_selection:
                continue

            # Loop over the variables we wish to evaluate, read them and
            # save them to the dataframe
            for name in variables_to_save:
                variables[name] = formulae[name].EvalInstance()
        
            try:
                truth_t_had, truth_t_lep  = get_truth_ttbar(variables)
                truth_param, costs = get_best_permutaiton(e, truth_t_had, truth_t_lep)
            except(KeyError):
                print('Skipping event due to problematic permutation!')
                continue 

            for i in xrange(len(costs)):
                variables["truth_p%d"%i] = 0.
                if truth_param == i:
                    variables["truth_p%d"%i] = 1.0

            # Construct the input the model expects
            X = get_model_input(variables)

            # Get the best permutation 
            Y = model.predict(X)
            best_perm = np.argmax(Y)
            permutation = get_permutation_vector(best_perm)

            # Perform the construction
            jets = get_jets_from_indices(e, permutation)
            t_had, t_lep = get_ttbar(jets, e)

            variables = append_reco_variables(variables, t_had, t_lep)
            variables = append_resolution_variables(variables, t_had, t_lep,truth_t_had, truth_t_lep, e)


            # Create the dataframe if we haven't done so already 
            if df is None:
                df = pd.DataFrame.from_dict(convert_elements_to_list(
                    variables
                ))
            else:
                # Append this new column to the dataframe  
                df = df.append(variables, ignore_index=True)
    #   
    print('Saving dataframe to file ',output_folder + output_file_name)
    df.to_csv(output_folder + output_file_name)

    # Save that data-frame
    print('Created data frame.')
    print('')
    print(df.head())
    print('')
    print('Number of events processed: ', chain.GetEntries())
    print('Number of events saved: ', len(df)) 


def main():
    # Construct the tchain of the input files
    mj.load_delphes_library()
    def tbar_is_leptonic(self,reco_event, truth_event):
        # the charge of the lepton is NEGATVIE since
        # tbar > W-b > l-nub

        # Evaluate which lepton's charge to evaluate
        if len(reco_event.el_pt) > 0:
            return reco_event.el_charge[0] < 0
        else:
            return reco_event.mu_charge[0] < 0


    # Run the conversion
    root_to_csv_jet_assignmet(
        input_file='/hepgpu3-data1/jrawling/deadcone_AT_tuples/4104070_mc16a/user.jrawling.15477193._000668.output.root',
        output_folder='/hepgpu3-data1/jrawling/deep_tops/csvs/',
        output_file_name='ttbar_processed.csv',
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
            'truth_had_top_pt':  "((MC_tbar_afterFSR_pt/1e3)*((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)+"
                                 "(MC_t_afterFSR_pt/1e3)*(1-((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)))",
            'truth_had_top_eta': "((MC_tbar_afterFSR_eta)*((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)+"
                                 "(MC_t_afterFSR_eta)*(1-((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)))",
            'truth_had_top_phi': "((MC_tbar_afterFSR_phi)*((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)+"
                                 "(MC_t_afterFSR_phi)*(1-((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)))",
            'truth_had_top_m':   "((MC_tbar_afterFSR_m/1e3)*((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)+"
                                 "(MC_t_afterFSR_m/1e3)*(1-((Length$(el_pt) > 0)*(el_charge[0] >= 0) + mu_charge[0] >= 0)))",

            'truth_lep_top_pt':  "((MC_tbar_afterFSR_pt/1e3)*((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)+"
                                 "(MC_t_afterFSR_pt/1e3)*(1-((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)))",
            'truth_lep_top_eta': "((MC_tbar_afterFSR_eta)*((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)+"
                                 "(MC_t_afterFSR_eta)*(1-((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)))",
            'truth_lep_top_phi': "((MC_tbar_afterFSR_phi)*((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)+"
                                 "(MC_t_afterFSR_phi)*(1-((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)))",
            'truth_lep_top_m':   "((MC_tbar_afterFSR_m/1e3)*((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)+"
                                 "(MC_t_afterFSR_m/1e3)*(1-((Length$(el_pt) > 0)*(el_charge[0] < 0) + mu_charge[0] < 0)))",
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
