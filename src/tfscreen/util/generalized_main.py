import sys 
import inspect
import argparse

def generalized_main(fcn,
                     argv=None,
                     manual_arg_defaults=None,
                     manual_arg_types=None,
                     manual_arg_nargs=None):
    """
    A generalized main function that constructs a command line argument parser
    and then parses arguments for any function. It infers argument defaults and
    types from the function signature. These inferences can be overwritten by 
    manual_arg_defaults and manual_arg_types. 

    Parameters
    ----------
    fcn: callable
        function to run.
    argv: iterable
        arguments to parse. if None, use sys.argv[1:]
    manual_arg_defaults : dict
        dictionary keying arguments to defaults that differ from the signature.
        The argument type is set to the type of the value specified. 
    manual_arg_types: dict
        dictionary keying arguments to types. This overrides the types inferred
        from the signature or manual_arg_defaults. 
    manual_arg_nargs: dict
        dictionary keying arguments to nargs. This overrides the nargs inferred
        from the signature or manual_arg_defaults. 
    """

    # Get command line arguments
    if argv is None:
        argv = sys.argv[1:]

    # Build dicts of manual arg types and defaults if not specified
    if manual_arg_types is None:
        manual_arg_types = {}

    if manual_arg_defaults is None:
        manual_arg_defaults = {}

    if manual_arg_nargs is None:
        manual_arg_nargs = {}

    # Build parser
    description = dict(inspect.getmembers(fcn))["__doc__"]
    parser = argparse.ArgumentParser(prog=f"{fcn.__name__}.py",
                                     description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Build parser arguments using signature of fcn
    param = inspect.signature(fcn).parameters
    for p in param:

        # Grab default and type the default for the parameter. If no default 
        # specified, make required. 
        if param[p].default is not param[p].empty:
            arg_type = type(param[p].default)
            default = param[p].default
            required = False
        else:
            arg_type = None
            default = None
            required = True
        
        # If the argument is in manual defaults, override what we got from the
        # function signature. 
        if p in manual_arg_defaults:
            arg_type = type(manual_arg_defaults[p])
            default = manual_arg_defaults[p]
            required = False
        
        # manual_arg_type takes precedence over any types inferred above. 
        if p in manual_arg_types:
            arg_type = manual_arg_types[p]

        # assume nargs is None unless the user overrides explicitly
        if p in manual_arg_nargs:
            nargs = manual_arg_nargs[p]
        else:
            nargs = None

        # Add the appropriate argument to the parser. 
        if required:
            parser.add_argument(p,type=arg_type,nargs=nargs)
        else:
            arg_name = f"--{p}"
            if arg_type is bool:
                if default is True:
                    parser.add_argument(arg_name,action="store_false")
                else:
                    parser.add_argument(arg_name,action="store_true")
            else:
                parser.add_argument(arg_name,type=arg_type,default=default,nargs=nargs)
                
    # Parse stats
    args = parser.parse_args(argv)

    # Call function with kwargs
    fcn(**args.__dict__)
