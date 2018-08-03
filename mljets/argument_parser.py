import json
import argparse

def setup_parser(arguments, description):

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Register the type of variables so they are readable from configs 
    parser.register('type','str',str)
    parser.register('type','float',float)
    parser.register('type','int',int)
    parser.register('type','bool',bool)
    parser.register('type','list',list)

    for key, val in arguments.items():
        parser.add_argument('-%s' % val['short'],
                            '--%s' % key,
                            type=val["type"],
                            help=val["help"],
                            default=val["default"])

    return parser


def read_params(parser):
    return vars(parser.parse_args())

def get_parameters(config_file,description=None, return_data=False):
    """
    Read in the arguments passed to a script as well reading in the arguments
    from a config file. 
    @Parameters
    config_file: a json configuration file to provide default configurations
    description [=None]: The text displayed when a script is called with -h  
    return_data: Whether the configuration file in a JSON friendly format  will 
                 be returned with the parameters 
    """

    with open(config_file) as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, description)
    parameters = read_params(parser)

    # The user might have set a different config file to be loaded 
    if parameters['config'] != "none":
        with open(parameters['config']) as data_file:
            data = json.load(data_file)

            for key, val in data.items():
                if key in parameters:
                    if parameters[key] != parser.get_default(key):
                        continue 
        
                parameters[key] = val["default"]

    if return_data:
        return parameters, data

    return parameters

