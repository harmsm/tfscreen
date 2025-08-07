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

# note: all polynomials go from lowest to highest degree (they are fed into 
# numpy.polynomial.Polynomial)

# polynomial describing growth rate vs iptg concentration (mM) for wild-type
# lac repressor in the absence of a marker or selection
wt_growth = [0.015,0.005]

# polynomials describing effect of marker production on growth rate as a 
# function of fractional operator saturation.
markers = {"kanR":[   0.006744862101985672,
                     -0.006651565351119809],
           "pheS":[   0.008640618397599201,
                      0.0021339347627602236]}

# polynomials describing effect of marker production on growth rate as a 
# function of fractional operator saturation.
selectors = {"kanR":[-0.01067521071786613,
                     -0.008994001136041463],
             "pheS":[-0.012493817129926034,
                      0.009408800930366923]}


wt_growth = np.polynomial.Polynomial(coef=wt_growth)

for m in markers:
    markers[m] = np.polynomial.Polynomial(coef=markers[m])

for s in selectors:
    selectors[s] = np.polynomial.Polynomial(coef=selectors[s])   