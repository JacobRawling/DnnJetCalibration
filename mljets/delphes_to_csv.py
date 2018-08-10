import ROOT as r
from tqdm import tqdm
import pandas as pd


def load_delphes_library():
    r.gSystem.Load("/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/Delphes/libDelphes")


class DelphsToCsv:

    def __init__(self,):
        pass

    def convert(self,
                input_file,
                output_folder,
                output_file_name,
                variables_to_save={},
                selection='1.0'
                ):
        # Construct the tchain of the input files
        chain = r.TChain('Delphes')
        chain.AddFile(input_file)


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
                if df == None:
                    df = pd.DataFrame(variables)
                else:
                    df = df.append(variables,ignore_index=True)
                    break
        # Save that data-frame  
        print('Created dataframe:')
        print(df.head())
        df.to_csv(output_folder + output_file_name)
