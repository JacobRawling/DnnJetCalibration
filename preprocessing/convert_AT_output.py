"""

"""
from __future__ import print_function
import mljets as mj
import ROOT as r
from tqdm import tqdm
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import glob

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
    t_had = jets[1] + jets[3] + jets[4]
    t_lep = jets[0] + e.W_lep
    return t_had, t_lep 

def cost_function(t_had, t_lep, truth_t_had, truth_t_lep):
    # mse = (t_hac  
    # # 8 DoF we're averaging over  
    # mse = mse/3.0/3.0
    mse = 0.0
    mse += (t_had.M() - truth_t_had.M() )**2 
    mse += (t_lep.M() - truth_t_lep.M() )**2 
    mse = mse/2.0
    return mse

def reco_cost_function(t_had, t_lep):
    # mse = (t_had.Eta() - truth_t_had.Eta())**2 
    # mse += (t_had.Phi() - truth_t_had.Phi())**2
    # # mse += (t_had.Pt() - truth_t_had.Pt())**2 
    # t_had_e = t_had.E() if t_had.E() > 0 else 1e-6
    # truth_t_had_e = truth_t_had.E() if truth_t_had.E() > 0 else 1e-6
    # mse += (t_had.M()/t_had_e - truth_t_had.M()/truth_t_had_e )**2 

    # mse += (t_lep.Eta() - truth_t_lep.Eta())**2 
    # mse += (t_lep.Phi() - truth_t_lep.Phi())**2
    # # mse += (t_lep.Pt() - truth_t_lep.Pt())**2 

    # t_lep_e = t_lep.E() if t_lep.E() > 0 else 1e-6
    # truth_t_lep_e = truth_t_lep.E() if truth_t_lep.E() > 0 else 1e-6
    # mse += (t_lep.M()/t_lep_e - truth_t_lep.M()/truth_t_lep_e )**2 

    # # 8 DoF we're averaging over  
    # mse = mse/3.0/3.0
    mse = 0.0
    mse += (t_had.M() - 172.5e3 )**2 
    mse += (t_lep.M() - 172.5e3 )**2 
    mse = mse/2.0
    return mse

def get_best_permutaiton(e, truth_t_had, truth_t_lep ):
    initial_permutation, cost, reco_cost = [0,1,2,3,4], {}, {}
    b_inital_permutation, l_inital_permutation = [0,1], [2,3,4]
    i,prev_two_assignments = 0, None
    t_lep_m_p, t_had_m_p = [],[]
    for b_i in multiset_permutations(b_inital_permutation):
        for l_i in multiset_permutations(l_inital_permutation):

            # Remove two fold degenrcy due W jet  
            last_two_assignments = l_i[-2:]
            if prev_two_assignments is not None:
                if set(last_two_assignments) == set(prev_two_assignments):
                    prev_two_assignments = last_two_assignments
                    continue 

            # 
            # print('permutation:', b_i+l_i)
            jets = get_jets_from_indices(e, b_i+l_i)
            # print('   jets:')
            # for j in jets:
            #     print( '    ', j.Pt(), j.Eta(), j.Phi(), j.E())

            t_had, t_lep = get_ttbar(jets, e)
            t_lep_m_p.append(t_lep.M())
            t_had_m_p.append(t_had.M())
            # print('   tops:')
            # print( '       ', t_had.Pt(), t_had.Eta(), t_had.Phi(), t_had.M()/1e3)
            # print( '       ', t_lep.Pt(), t_lep.Eta(), t_lep.Phi(), t_lep.M()/1e3)

            # Reconstruct the top quark 
            L = cost_function( 
                            t_had, t_lep,
                            truth_t_had, truth_t_lep 
                        )
            cost[i] = L
            reco_cost[i] = reco_cost_function( t_had, t_lep)

            # print('  Cost: ', L)
            prev_two_assignments = last_two_assignments
            i += 1

    # print(cost)
    min_index =  min(cost, key=cost.get)
    # print('t_had_m_p: ', t_had_m_p)
    # print('t_lep_m_p: ', t_lep_m_p)
    # print('cost: ', cost)
    # print('')
    # print('')
    # print('')
    return min_index, cost.keys(),t_lep_m_p, t_had_m_p, reco_cost

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
    truth_t_had, truth_t_lep = r.TLorentzVector(), r.TLorentzVector()
    
    truth_t_had.SetPtEtaPhiM(
            variables['truth_had_top_pt'],
            variables['truth_had_top_eta'],
            variables['truth_had_top_phi'],
            variables['truth_had_top_m'],
        )
    truth_t_lep.SetPtEtaPhiM(
            variables['truth_lep_top_pt'],
            variables['truth_lep_top_eta'],
            variables['truth_lep_top_phi'],
            variables['truth_lep_top_m'],
        )

    return truth_t_had, truth_t_lep 

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
    input_files = glob.glob(input_file)
    for f in input_files:
        chain.AddFile(f)

    if friend_tuple_name is not None:
        friend_chain = r.TChain(friend_tuple_name)
        for f in input_files:
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
    n = 0
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
                formulae[name].GetNdata()
            
            try:
                truth_t_had, truth_t_lep  = get_truth_ttbar(variables)
                truth_param, costs, t_had_m_p, t_lep_m_p,reco_cost = get_best_permutaiton(e, truth_t_had, truth_t_lep)
                delta_t_had_m_p, delta_t_lep_m_p = np.array(t_had_m_p), np.array(t_lep_m_p)
                min_reco_index =  min(reco_cost, key=reco_cost.get)

            except(KeyError):
                print('Skipping event due to problematic permutation!')
                continue 

            for i in xrange(len(costs)):
                variables["truth_p%d"%i] = 0.
                if truth_param == i:
                    variables["truth_p%d"%i] = 1.0
                variables["t_had_m_p_%d"%i] = t_had_m_p[i]
                variables["t_lep_m_p_%d"%i] = t_lep_m_p[i]
                variables["delta_t_lep_%d"] = delta_t_lep_m_p
                variables["delta_t_had_%d"] = delta_t_had_m_p

            variables["best_lep_m_perm"] = np.argmin(delta_t_lep_m_p)
            variables["best_had_m_perm"] = np.argmin(delta_t_had_m_p)
                         
            variables["truth_param"] = truth_param#
            variables["best_reco_param"] = min_reco_index

            # Create the dataframe if we haven't done so already 
            if df is None:
                df = pd.DataFrame.from_dict(convert_elements_to_list(
                    variables
                ))
            else:
                # Append this new column to the dataframe  
                df = df.append(variables, ignore_index=True)
            n+=1
    #
    print('Creating delta R variables')
    for i in range(5):
        df['dR_Wj%d'%i] = np.sqrt( np.square(df['jet_eta_%d'%i] - df['W_lep_eta']) + np.square(df['jet_phi_%d'%i] - df['W_lep_phi']) )
        df['dR_Lj%d'%i] = np.sqrt( np.square(df['jet_eta_%d'%i] - df['lep_eta']) + np.square(df['jet_phi_%d'%i] - df['lep_phi']) )
        df['j%d_met_dPhi'%i] = np.sqrt( np.square(df['met_phi'] - df['jet_phi_%d'%i])  )

        for j in range(i+1,5):
            df['dR_j%dj%d'%(i,j)] = np.sqrt( np.square(df['jet_eta_%d'%i] - df['jet_eta_%d'%j]) + np.square(df['jet_phi_%d'%i] - df['jet_eta_%d'%j]) )

    df['closest_to_W']  =np.argmin(df[['dR_Wj%d'%i for i in range(5)]].values,axis=1)
    df['n_bjets'] = np.sum(df[['jet_isBjet_0','jet_isBjet_1','jet_isBjet_2','jet_isBjet_3','jet_isBjet_4']].values,axis=1)
    df['met_l_dPhi'] = np.sqrt( np.square(df['met_phi'] - df['lep_phi'])  )

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
    infiles = glob.glob('/hepgpu3-data1/jrawling/deadcone_AT_tuples/4104070_mc16a/*output.root')
    print('Processing %d files'%len(infiles))
    for i,f in enumerate(infiles):
        print('Processing file: ', f)
        
        root_to_csv_jet_assignmet(
            input_file=f,
            output_folder='/hepgpu3-data1/jrawling/deep_tops/csvs_test/',
            output_file_name='ttbar_%d.csv'%i,
            variables_to_save={
                # Reco level
                'n_jets': 'Length$(jet_pt)',
                'jet_pt_0':  'jet_pt[0]/1e3',
                'jet_eta_0': 'jet_eta[0]',
                'jet_phi_0': 'jet_phi[0]',
                'jet_e_0':   'jet_e[0]/1e3',
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

                'W_lep_pt': 'W_lep.Pt()/1e3',
                'W_lep_eta': 'W_lep.Eta()',
                'W_lep_phi': 'W_lep.Phi()',
                'W_lep_m': 'W_lep.M()/1e3',

                'lep_pt':  'Alt$(el_pt[0]/1e3,Alt$(mu_pt[0]/1e3,0))',
                'lep_eta': 'Alt$(el_eta[0],Alt$(mu_eta[0],0))',
                'lep_phi': 'Alt$(el_phi[0],Alt$(mu_phi[0],0))',

                'met': 'met_met',
                'met_phi': 'met_phi',

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
            selection='(ejets_particle+mujets_particle)',
            tuple_name='particleLevel',
            friend_tuple_name='truth',
            major_index="(runNumber<<13)",
            minor_index="eventNumber"
            )


main()
