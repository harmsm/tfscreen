from tfscreen.analysis.hierarchical.tensor_manager import TensorManager
from tfscreen.analysis.hierarchical.growth_model.model import GrowthModelData

def df_to_tensors(df):
    """
    Take an input dataframe and generate 4D tensors for model regression. 
    
    The goal is to generate tensors with a shape going from lowest to highest
    number of values: (replicate,time,treatment,genotypes). 
    """

    # -------------------------------------------------------------------------
    # Prep dataframe for processing 
    # -------------------------------------------------------------------------
    
    tm = TensorManager(df)
        
    tm.add_parameter_map("ln_cfu0",["replicate","condition_pre","genotype"])    
    tm.add_parameter_map("A",["genotype"])
    tm.add_parameter_map("dk_geno",["genotype"])
    tm.add_parameter_map("theta",["genotype","titrant_name","titrant_conc"])
    tm.add_parameter_map("titrant",["titrant_name","titrant_conc"])
    tm.add_parameter_map("theta_group",["genotype","titrant_name"])
    tm.add_parameter_map("replicate",["replicate"])

    tm.add_condition_map()
    tm.add_data_tensor("titrant_conc")

    tm.create_tensors()

    data = tm.build_dataclass(GrowthModelData)

    return tm, data

    # # Build model data class
    # gmd = GrowthModelData(**data_dict)

    # gmp = GrowthModelParameters(tensor_labels,
    #                             parameter_indexers)
                      
    # return df, gmd, gmp
    
