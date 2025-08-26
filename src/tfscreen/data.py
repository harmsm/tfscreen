"""
Constants and lookup tables for codon and amino acid data used in tfscreen.
"""

import numpy as np

codon_to_aa = {
    'ttt': 'F', 'ttc': 'F', 'tta': 'L', 'ttg': 'L',
    'ctt': 'L', 'ctc': 'L', 'cta': 'L', 'ctg': 'L',
    'att': 'I', 'atc': 'I', 'ata': 'I', 'atg': 'M',
    'gtt': 'V', 'gtc': 'V', 'gta': 'V', 'gtg': 'V',
    'tct': 'S', 'tcc': 'S', 'tca': 'S', 'tcg': 'S',
    'cct': 'P', 'ccc': 'P', 'cca': 'P', 'ccg': 'P',
    'act': 'T', 'acc': 'T', 'aca': 'T', 'acg': 'T',
    'gct': 'A', 'gcc': 'A', 'gca': 'A', 'gcg': 'A',
    'tat': 'Y', 'tac': 'Y', 'taa': '*', 'tag': '*',
    'cat': 'H', 'cac': 'H', 'caa': 'Q', 'cag': 'Q',
    'aat': 'N', 'aac': 'N', 'aaa': 'K', 'aag': 'K',
    'gat': 'D', 'gac': 'D', 'gaa': 'E', 'gag': 'E',
    'tgt': 'C', 'tgc': 'C', 'tga': '*', 'tgg': 'W',
    'cgt': 'R', 'cgc': 'R', 'cga': 'R', 'cgg': 'R',
    'agt': 'S', 'agc': 'S', 'aga': 'R', 'agg': 'R',
    'ggt': 'G', 'ggc': 'G', 'gga': 'G', 'ggg': 'G'
}

degen_base_specifier = {
    "a":"a",
    "c":"c",
    "g":"g",
    "t":"t",
    "r":"ag",
    "y":"ct",
    "m":"ac",
    "s":"cg",
    "w":"at",
    "h":"act",
    "b":"cgt",
    "v":"acg",
    "d":"agt",
    "n":"acgt"
}

