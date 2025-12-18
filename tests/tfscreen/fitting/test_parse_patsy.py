
import pytest
import pandas as pd
import numpy as np
from tfscreen.fitting.parse_patsy import parse_patsy

def test_parse_patsy_simple():
    df = pd.DataFrame({"color": ["red", "blue"], "size": [1, 2]})
    
    # Patsy often generates names like "beta[T.red]" if formula is "y ~ C(color)"
    # If using custom terms, it might look different.
    # The code expects `param[specifier]`.
    
    patsy_param_names = [
        "beta[T.red]",
        "beta[T.blue]"
    ]
    
    # model_terms maps "beta" -> "beta" usually?
    # Code: patsy_to_pretty = dict((value,key) for key, value in model_terms.items())
    # So if model_terms = {"beta": "beta"}, patsy_to_pretty["beta"] = "beta".
    model_terms = {"beta": "beta"}
    
    # factor_terms maps param_class ("beta") to columns ("color")
    factor_terms = {"beta": "color"}
    
    res = parse_patsy(df, patsy_param_names, model_terms, factor_terms)
    
    assert len(res) == 2
    assert "color" in res.columns
    assert res.iloc[0]["param_class"] == "beta"
    assert res.iloc[0]["color"] == "red" # Coerced to string
    assert res.iloc[1]["color"] == "blue"

def test_parse_patsy_numeric_coercion():
    df = pd.DataFrame({"size": [1, 2]})
    patsy_param_names = ["gamma[1]", "gamma[2]"]
    model_terms = {"gamma": "gamma"}
    factor_terms = {"gamma": "size"}
    
    res = parse_patsy(df, patsy_param_names, model_terms, factor_terms)
    
    # Should be coerced to int because df["size"] is int
    assert res.iloc[0]["size"] == 1
    assert isinstance(res.iloc[0]["size"], (int, np.integer))

def test_parse_patsy_tuple_factors():
    df = pd.DataFrame({"color": ["red"], "temp": [10]})
    patsy_param_names = ["k[red, 10]"] # patsy interaction term often: k[T.red:10] or k[red, 10]?
    # Code handles specifier that looks like tuple? 
    # "If this looks like a tuple, split it into individual values"
    # But patsy output for interaction is usually "A:B". 
    # The code specifically checks for `(` and `)`.
    # Maybe custom coding? Or how patsy formats tuples.
    # "If specifier[0] == '(' ... "
    
    patsy_param_names = ["k[(red, 10)]"]
    model_terms = {"k": "k"}
    factor_terms = {"k": ["color", "temp"]}
    
    res = parse_patsy(df, patsy_param_names, model_terms, factor_terms)
    
    assert res.iloc[0]["color"] == "red"
    assert res.iloc[0]["temp"] == 10

def test_parse_patsy_errors():
    df = pd.DataFrame({"color": ["red"]})
    patsy_param_names = ["k[red]"]
    model_terms = {"k": "k"}
    factor_terms = {"k": ["color", "shape"]} # Mismatch length
    
    with pytest.raises(RuntimeError, match="problem parsing the patsy"):
        parse_patsy(df, patsy_param_names, model_terms, factor_terms)

def test_parse_patsy_coercion_error():
    df = pd.DataFrame({"val": [1]}) # int
    patsy_param_names = ["k[foo]"] # 'foo' cannot be int
    model_terms = {"k": "k"}
    factor_terms = {"k": "val"}
    
    with pytest.raises(RuntimeError, match="problem coercing the factor"):
        parse_patsy(df, patsy_param_names, model_terms, factor_terms)
