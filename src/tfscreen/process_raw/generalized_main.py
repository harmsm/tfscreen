import sys 
import inspect
import argparse

def generalized_main(fcn,argv=None,optional_arg_types=None):
    """
    A generalized main function that constructs a command line argument parser
    and then parses arguments for any function.

    Parameters
    ----------
    fcn: callable
        function to run.
    argv: iterable
        arguments to parse. if none, use sys.argv[1:]
    optional_arg_types: dict
        dictionary of arg types for arguments with None as their default in the
        function. If an argument is not in optional_arg_types, the default is to
        treat argument as an int.
    """

    # Get command line arguments
    if argv is None:
        argv = sys.argv[1:]

    if optional_arg_types is None:
        optional_arg_types = {}

    # Build parser
    description = dict(inspect.getmembers(fcn))["__doc__"]
    parser = argparse.ArgumentParser(prog=f"{fcn.__name__}.py",
                                     description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Build parser arguments using signature of fcn
    param = inspect.signature(fcn).parameters
    for p in param:

        # If no default specified, make required
        if param[p].default is param[p].empty:
            parser.add_argument(p)

        # If default specified, make optional
        else:

            # For type == None args, parse as integer
            if param[p].default is None:
                try:
                    arg_type = optional_arg_types[p]
                except KeyError:
                    arg_type = int

            else:
                arg_type = type(param[p].default)

            parser.add_argument(f"--{p}",
                                type=arg_type,
                                default=param[p].default)

    # Parse stats
    args = parser.parse_args(argv)

    # Call function with kwargs
    fcn(**args.__dict__)
