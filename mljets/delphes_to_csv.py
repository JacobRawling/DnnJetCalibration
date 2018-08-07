import ROOT as r 
import tqdm as td 
r.gSystem.Load("/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/Delphes/libDelphes")

class DelphsToCsv:
    def __init__( 
            input_file,
            output_folder,
            output_file_name,
            variables_to_save={
                'e.Jet[0].PT': ''

            }
        ):


    def convert(self):
        # Construct the tchain of the input files
        chain = r.TChain('Delphes')
        chain.AddFile(input_file)

        # Create the pandas dataframe that will store the thing 

        # Create the TTreeForumlas based on the variables_to_save to allow us
        # to easily add inputs in a TTree::Draw style


        # Iterate over the tree and 
        for e in tqdm(chain):

            # Loop over the variables we wish to evaluate, read them and 
            # save them to the datafrem 
            if isinstance(e.Jet[0], r.Jet):
                print(type(e.Jet[0]))
                print(e.Jet[0].PT)

        # Save that data-frame 
