import argparse
import inspect
import os
import re
import sys

from tfscreen.util import provenance


def _split_docstring(doc):
    """
    Split a numpy-style docstring into its description and per-parameter help.

    Returns
    -------
    description : str
        Everything before the ``Parameters`` section, dedented.
    param_help : dict
        Parameter name -> its description, joined onto one line.
    """
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    lines = doc.splitlines()

    def _is_header(i, name=None):
        if i + 1 >= len(lines):
            return False
        underline = lines[i + 1].strip()
        if not underline or set(underline) != {"-"}:
            return False
        return name is None or lines[i].strip() == name

    start = next((i for i in range(len(lines)) if _is_header(i, "Parameters")), None)
    if start is None:
        return doc.strip(), {}
    description = "\n".join(lines[:start]).strip()

    param_help = {}
    current = None
    for i in range(start + 2, len(lines)):
        if _is_header(i):
            break
        line = lines[i]
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            # "name : type" or "name" (numpydoc allows several names "a, b")
            names = line.split(":")[0]
            current = [n.strip() for n in names.split(",") if n.strip()]
            for n in current:
                param_help[n] = ""
        elif current is not None:
            for n in current:
                param_help[n] = (param_help[n] + " " + line.strip()).strip()
    return description, param_help


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep the description's layout; wrap argument help normally."""


class ParserCaptured(Exception):
    """Raised instead of running the command while ``capture_parsers`` is on."""

    def __init__(self, parser):
        super().__init__(parser.prog)
        self.parser = parser


# When True, generalized_main builds the parser and raises ParserCaptured
# instead of parsing and running (used to generate the CLI reference).
_CAPTURE_PARSER = False


def capture_parser(main):
    """
    The argparse parser a ``tfs-*`` ``main()`` builds, without running it.

    Parameters
    ----------
    main : callable
        An entry point's ``main`` function (one that calls generalized_main).

    Returns
    -------
    argparse.ArgumentParser
    """
    global _CAPTURE_PARSER
    _CAPTURE_PARSER = True
    try:
        main()
    except ParserCaptured as e:
        return e.parser
    finally:
        _CAPTURE_PARSER = False
    raise RuntimeError(f"{main} did not call generalized_main")


def _help_text(param_help, name, default, show_default):
    text = param_help.get(name, "")
    # Reduce double-backtick literals to plain text for the terminal.
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = text.replace("%", "%%")
    if show_default and "default" not in text.lower():
        text = f"{text} (default: {default!r})" if text else f"(default: {default!r})"
    return text or None


def generalized_main(fcn,
                     argv=None,
                     manual_arg_defaults=None,
                     manual_arg_types=None,
                     manual_arg_nargs=None,
                     prog=None,
                     write_provenance=True):
    """
    Build a command-line parser from a function's signature and run it.

    Parameters without a default become positional arguments; parameters with
    a default become ``--name`` flags of the default's type. A bool that
    defaults to False is a ``--name`` switch; one that defaults to True takes
    ``--no_name`` to turn it off (``--name`` is accepted and does nothing). A
    flag whose default is None reads a string unless ``manual_arg_types``
    gives its type. The function's numpy-style docstring supplies the help:
    its text before ``Parameters`` is the description, and each parameter's
    entry is that argument's help.

    Before the function runs, the run's provenance (tfscreen version, git
    commit, command line) is printed to stderr. If the function takes an
    ``out_prefix`` argument, it is also written to
    ``{out_prefix}_provenance.json``.

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
    prog : str, optional
        program name shown in usage. Default: the name the command was run as
        (``sys.argv[0]``) when ``argv`` is None, else the function name.
    write_provenance : bool
        print the provenance and write ``{out_prefix}_provenance.json``.

    Returns
    -------
    None
        Always None, whatever ``fcn`` returns: console scripts run
        ``sys.exit(main())``, and any other value would end the process with
        status 1 and print the value.
    """

    command_argv = list(sys.argv) if argv is None else [prog or fcn.__name__, *argv]
    if prog is None:
        prog = os.path.basename(sys.argv[0]) if argv is None else fcn.__name__
    if argv is None:
        argv = sys.argv[1:]

    manual_arg_types = manual_arg_types or {}
    manual_arg_defaults = manual_arg_defaults or {}
    manual_arg_nargs = manual_arg_nargs or {}

    description, param_help = _split_docstring(fcn.__doc__)
    description = re.sub(r"``([^`]+)``", r"\1", description)
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=_HelpFormatter)

    param = inspect.signature(fcn).parameters
    for p in param:

        if param[p].default is not param[p].empty:
            default = param[p].default
            arg_type = type(default)
            required = False
        else:
            default = None
            arg_type = None
            required = True

        if p in manual_arg_defaults:
            default = manual_arg_defaults[p]
            arg_type = type(default)
            required = False

        if p in manual_arg_types:
            arg_type = manual_arg_types[p]
        elif arg_type is type(None):
            arg_type = str

        nargs = manual_arg_nargs.get(p, None)

        if required:
            parser.add_argument(p, type=arg_type, nargs=nargs,
                                help=_help_text(param_help, p, None, False))
            continue

        arg_name = f"--{p}"
        if arg_type is bool:
            if default is True:
                text = _help_text(param_help, p, None, False)
                parser.add_argument(f"--no_{p}", dest=p, action="store_false",
                                    help=(f"Turn off (on by default): {text}"
                                          if text else f"turn off {p}"))
                parser.add_argument(arg_name, dest=p, action="store_true",
                                    help=argparse.SUPPRESS)
                parser.set_defaults(**{p: True})
            else:
                parser.add_argument(arg_name, action="store_true",
                                    help=_help_text(param_help, p, None, False))
            continue

        # With nargs, the type is the element type, not the list itself.
        if nargs is not None and arg_type in (list, tuple):
            if default is not None and len(default) > 0:
                arg_type = type(default[0])
            else:
                arg_type = str

        parser.add_argument(arg_name, type=arg_type, default=default, nargs=nargs,
                            help=_help_text(param_help, p, default, True))

    if _CAPTURE_PARSER:
        raise ParserCaptured(parser)

    args = parser.parse_args(argv)
    kwargs = args.__dict__

    if write_provenance:
        provenance.set_command(command_argv)
        prov = provenance.get_provenance()
        provenance.print_provenance(prov)
        out_prefix = kwargs.get("out_prefix", None)
        if isinstance(out_prefix, str) and out_prefix:
            provenance.write_provenance(f"{out_prefix}_provenance.json", prov)

    fcn(**kwargs)
    return None
