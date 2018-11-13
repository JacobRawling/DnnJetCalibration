from __future__ import print_function
import ROOT as r
from tqdm import tqdm
import pandas as pd

def load_delphes_library():
    """
    ToDo: Move this to a configurable path in the configs folder. 
    """
    r.gSystem.Load("/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/Delphes/libDelphes")


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


def root_to_csv(
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
    chain.AddFile(input_file)

    if friend_tuple_name is not None:
        friend_chain = r.TChain(friend_tuple_name)
        friend_chain.AddFile(input_file)
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

