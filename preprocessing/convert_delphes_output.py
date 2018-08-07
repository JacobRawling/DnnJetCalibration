"""

"""

# import mljets as mj 
import ROOT as r 
from tqdm import tqdm
r.gSystem.Load("/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/Delphes/libDelphes")

def main():
     # Construct the tchain of the input files
    input_file = '/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/dijet_2/Events/run_03/tag_1_delphes_events.root'
    chain = r.TChain('Delphes')
    chain.AddFile(input_file)

    with tqdm(total=chain.GetEntries()) as progress_bar:
        for e in chain:
            progress_bar.update(1)



main()