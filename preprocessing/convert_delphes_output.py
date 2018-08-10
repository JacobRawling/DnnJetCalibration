"""

"""

import mljets as mj
import ROOT as r
from tqdm import tqdm


def main():
    # Construct the tchain of the input files
    mj.load_delphes_library()

    # Set up the conversion tool
    delphes_convertor = mj.DelphsToCsv()

    # Run the conversion
    delphes_convertor.convert(
        input_file='/hepgpu3-data1/jrawling/MG5_aMC_v2_5_5/dijet_2/Events/run_03/tag_1_delphes_events.root',
        output_folder='/hepgpu3-data1/jrawling/deep_jets/csvs',
        output_file_name='tag_1_delphes_jets.csv',
        variables_to_save={
            'jet_pt_0': 'Jet.PT[0]'
            },  
        selection='Jet_size>=1'
        )


main()
