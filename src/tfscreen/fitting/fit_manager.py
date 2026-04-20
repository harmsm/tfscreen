import numpy as np
import pandas as pd
from scipy.special import expit, logit
from typing import Dict, Any


class FitManager:
    """
    Manages the parameters, data, and transformations for a regression fit.

    This class handles parameter transformations, bounds, fixed parameters,
    and predictions based on a design matrix.

    Parameters
    ----------
    y_obs : np.ndarray
        The observed values.
    y_std : np.ndarray
        The standard errors on the observed values.
    X : np.ndarray
        The design matrix.
    param_df : pd.DataFrame
        A DataFrame describing the parameters. It must contain a "guess"
        column. Optional columns include "lower_bounds", "upper_bounds",
        "censored", "fixed", "scale_mu", "scale_sigma", or "transform".
        The transform column can have the values 'scale' or 'logistic', 
        indicating the parameter should be scaled
        (v - scale_mu)/scale_sigma or logit'd. Any other values in this
        column are ignored the parameter is not scaled. 

    Notes
    -----
    This class assumes the user inputs (guesses, bounds) are all in the
    original, un-transformed space. The attributes guesses_transformed, 
    lower_bounds_transformed, and upper_bounds_transformed provide
    access to these after any transformations specified in the input 
    dataframe.
    """

    _LOGISTIC_STABILITY_BOUND = 20
    
    def __init__(self,
                 y_obs: np.ndarray,
                 y_std: np.ndarray,
                 X: np.ndarray,
                 param_df: pd.DataFrame):

        # Make private copies of inputs to ensure internal consistency.
        self._y_obs = y_obs.copy()
        self._y_std = y_std.copy()
        self._X = X.copy()
        self._param_df = param_df.copy()

        # Get number of observations and parameters
        self.num_obs = len(self.y_obs)
        self.num_params = len(self.param_df)

        # Validate and populate the parameter dataframe with default columns.
        self._setup_param_df()

        # --- Setup Transformations ---
        self._is_logistic = (self._param_df["transform"] == "logistic").to_numpy()
        self._is_scale = (self._param_df["transform"] == "scale").to_numpy()

        self._scale_mu = self._param_df["scale_mu"].to_numpy()
        self._scale_sigma = self._param_df["scale_sigma"].to_numpy()

        # For numerical stability, ensure sigma is positive.
        if np.any(self._scale_sigma[self._is_scale] <= 0):
            raise ValueError("Values in 'scale_sigma' must be positive.")
            
        self._model_func = None
        
    def set_model_func(self, func):
        """
        Set a custom model function for prediction.
        
        Parameters
        ----------
        func : callable
            A function with signature `func(params, X)` that returns predicted y.
            If set, this overrides the default linear prediction `y = X @ params`.
        """
        self._model_func = func
    
    def __repr__(self) -> str:
        """
        Provides the official string representation of the FitManager.
        """
        return f"<FitManager(num_obs={self.num_obs}, num_params={self.num_params})>"

    def __str__(self) -> str:
        """
        Provides a user-friendly string summary of the FitManager.
        """

        # Make sure these are strings before printing
        param_names = [f"{v}" for v in self._param_df.index.tolist()]
        
        return (
            f"FitManager Summary\n"
            f"------------------\n"
            f"Observations: {self.num_obs}\n"
            f"Parameters:   {self.num_params}\n"
            f"  - {', '.join(param_names)}"
        )

    def _repr_html_(self) -> str:
        """
        Provides a rich HTML representation for Jupyter notebooks.
        """
        # Using .to_html() on the dataframe is a very convenient way to format it.
        param_table_html = self._param_df.to_html()
        
        html = (
            f"<h3>FitManager</h3>"
            f"<b>Observations:</b> {self.num_obs}<br>"
            f"<b>Parameters:</b> {self.num_params}"
            f"<h4>Parameter Details</h4>"
            f"{param_table_html}"
        )
        return html
    def _setup_param_df(self):
        """
        Validates required columns and populates the parameter dataframe
        with default values for optional columns.
        """
        # Define expected columns and their default values/setup logic
        column_defaults: Dict[str, Dict[str, Any]] = {
            "guess": {"required": True},
            "lower_bounds": {"default": -np.inf},
            "upper_bounds": {"default": np.inf},
            "transform": {"default": None},
            "censored": {"default": False},
            "fixed": {"default": False},
            "scale_mu": {"default": 0},
            "scale_sigma": {"default": 1},
        }

        if "fixed" in self._param_df.columns:
            if not self._param_df["fixed"].dtype == bool:
                self._param_df["fixed"] = self._param_df["fixed"].fillna(0).astype(bool)

        for col, settings in column_defaults.items():
            if col not in self._param_df.columns:
                if settings.get("required", False):
                    raise ValueError(f"param_df must have a column named '{col}'.")
                self._param_df[col] = settings["default"]

    def transform(self, v: np.ndarray) -> np.ndarray:
        """
        Transforms parameters according to the rules set at initialization.

        Parameters
        ----------
        v : np.ndarray
            Parameter vector in the original (back_transformed) space.

        Returns
        -------
        np.ndarray
            Parameter vector in the transformed space.
        """
        out = v.copy()
        out[self._is_scale] = self._to_scale(v, mask=self._is_scale)
        out[self._is_logistic] = self._to_logistic(v[self._is_logistic])
        return out

    def back_transform(self, v_transformed: np.ndarray) -> np.ndarray:
        """
        Back transforms parameters to their original space.

        Parameters
        ----------
        v_transformed : np.ndarray
            Parameter vector in the transformed space.

        Returns
        -------
        np.ndarray
            Parameter vector in the original (un-transformed) space.
        """
        out = v_transformed.copy()
        out[self._is_scale] = self._from_scale(v_transformed, mask=self._is_scale)
        out[self._is_logistic] = self._from_logistic(v_transformed[self._is_logistic])
        out[self.is_fixed] = self._param_df.loc[self.is_fixed,"guess"]
        return out

    def back_transform_std_err(self, v_transformed: np.ndarray, std_err_transformed: np.ndarray) -> np.ndarray:
        """
        Back-transforms standard errors from the transformed space to the
        original space.

        This method uses the delta method (a first-order Taylor expansion)
        to propagate the uncertainty.

        Parameters
        ----------
        v_transformed : np.ndarray
            The parameter estimates in the transformed space.
        std_err_transformed : np.ndarray
            The standard errors of the parameters in the transformed space.

        Returns
        -------
        np.ndarray
            The standard errors in the original un-transformed space.
        """
        out_std = std_err_transformed.copy()

        # Handle scale transformation
        if np.any(self._is_scale):
            out_std[self._is_scale] = self._scale_sigma[self._is_scale] * std_err_transformed[self._is_scale]

        # Handle logistic transformation
        if np.any(self._is_logistic):

            v_t_logistic = v_transformed[self._is_logistic]
            
            # Derivative of the back-transformation (expit)
            # derivative = expit(v_t) * (1 - expit(v_t))
            derivative = self._from_logistic(v_t_logistic) * (1 - self._from_logistic(v_t_logistic))
            
            out_std[self._is_logistic] = derivative * std_err_transformed[self._is_logistic]
            
        return out_std
    
    def predict(self, v: np.ndarray) -> np.ndarray:
        """
        Predicts y-values from an un-transformed parameter vector.

        This method assumes 'v' contains values for all parameters, but enforces
        that fixed parameters use their initial guess values.

        Parameters
        ----------
        v : np.ndarray
            The un-transformed parameter vector.

        Returns
        -------
        np.ndarray
            The predicted y-values.
        """
        real_v = v.copy()
        real_v[self.is_fixed] = self.guesses[self.is_fixed]
        
        if self._model_func is not None:
            return self._model_func(real_v, self._X)
            
        return self._X @ real_v

    def predict_from_transformed(self, v_transformed: np.ndarray) -> np.ndarray:
        """
        Back-transforms a parameter vector and then predicts y-values.

        Parameters
        ----------
        v_transformed : np.ndarray
            The transformed parameter vector.

        Returns
        -------
        np.ndarray
            The predicted y-values.
        """
        real_v = self.back_transform(v_transformed)
        real_v[self.is_fixed] = self.guesses[self.is_fixed]
        
        if self._model_func is not None:
            return self._model_func(real_v, self._X)
            
        return self._X @ real_v

    def _from_logistic(self, v: np.ndarray, mask=None) -> np.ndarray:
        """
        Logistic transformation (sigmoid).

        Parameters
        ----------
        v : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed values.
        """

        if mask is not None:
            v = v[mask] 
        return expit(v)

    def _to_logistic(self, v: np.ndarray,mask=None) -> np.ndarray:
        """
        Inverse logistic transformation (logit).

        Note
        ----
        The logit function log(p / (1-p)) is undefined at p=0 and p=1.
        The input is clipped to be within (epsilon, 1-epsilon) for
        numerical stability.

        Parameters
        ----------
        v : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed values.
        """

        if mask is not None:
            v = v[mask]

        epsilon = np.finfo(v.dtype).eps
        v_clipped = np.clip(v, epsilon, 1 - epsilon)
  
        return logit(v_clipped)

    def _to_scale(self, v: np.ndarray, mask=None) -> np.ndarray:
        """
        Applies scaling: (v - mu) / sigma.

        Parameters
        ----------
        v : np.ndarray
            Input values.

        Returns
        -------
        np.ndarray
            Transformed values.
        """
    
        if mask is not None:
            return (v[mask] - self._scale_mu[mask]) / self._scale_sigma[mask]
        
        return (v - self._scale_mu) / self._scale_sigma
        

    def _from_scale(self, v: np.ndarray,mask=None) -> np.ndarray:
        """
        Reverses scaling: v * sigma + mu.

        Parameters
        ----------
        v : np.ndarray
            Input values (transformed).

        Returns
        -------
        np.ndarray
            Un-transformed values.
        """

        if mask is not None:
            return v[mask] * self._scale_sigma[mask] + self._scale_mu[mask]

        return v * self._scale_sigma + self._scale_mu

    @property
    def y_obs(self) -> np.ndarray:
        """np.ndarray: The observed values."""
        return self._y_obs

    @property
    def y_std(self) -> np.ndarray:
        """np.ndarray: The standard errors on the observed values."""
        return self._y_std

    @property
    def X(self) -> np.ndarray:
        """np.ndarray: The design matrix."""
        return self._X

    @property
    def param_df(self) -> pd.DataFrame:
        """pd.DataFrame: The DataFrame describing the parameters."""
        return self._param_df

    @property
    def lower_bounds(self) -> np.ndarray:
        """np.ndarray: The lower bounds for each parameter."""
        return self._param_df["lower_bounds"].to_numpy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """np.ndarray: The upper bounds for each parameter."""
        return self._param_df["upper_bounds"].to_numpy()

    @property
    def lower_bounds_transformed(self) -> np.ndarray:
        """
        np.ndarray: The lower bounds in the transformed space.
        """

        bounds = self.lower_bounds
        transformed_bounds = bounds.copy()
        finite_mask = np.isfinite(bounds)

        scale_and_finite = self._is_scale & finite_mask
        logistic_and_finite = self._is_logistic & finite_mask

        # If any values are not infinite
        if np.any(scale_and_finite):
            transformed_bounds[scale_and_finite] = self._to_scale(
                bounds,
                mask=scale_and_finite
            )

        # If any values are not infinite
        if np.any(logistic_and_finite):
            transformed_bounds[logistic_and_finite] = self._to_logistic(
                bounds,
                mask=logistic_and_finite
            )

        # Enforce bound of -15 on logistic for numerical stability
        logistic_and_not_set = self._is_logistic & ~finite_mask
        if np.any(logistic_and_not_set):
            transformed_bounds[logistic_and_not_set] = -self._LOGISTIC_STABILITY_BOUND
        
        return transformed_bounds

    @property
    def upper_bounds_transformed(self) -> np.ndarray:
        """
        np.ndarray: The upper bounds in the transformed space.
        """
        bounds = self.upper_bounds
        transformed_bounds = bounds.copy()
        finite_mask = np.isfinite(bounds)

        scale_and_finite = self._is_scale & finite_mask
        logistic_and_finite = self._is_logistic & finite_mask

        if np.any(scale_and_finite):
            transformed_bounds[scale_and_finite] = self._to_scale(
                bounds,
                mask=scale_and_finite
            )

        if np.any(logistic_and_finite):
            transformed_bounds[logistic_and_finite] = self._to_logistic(
                bounds,
                mask=logistic_and_finite
            )

        # Enforce bound of +15 on logistic for numerical stability
        logistic_and_not_set = self._is_logistic & ~finite_mask
        if np.any(logistic_and_not_set):
            transformed_bounds[logistic_and_not_set] = self._LOGISTIC_STABILITY_BOUND
        
        return transformed_bounds

    @property
    def censored(self) -> np.ndarray:
        """np.ndarray: A boolean array indicating if a parameter is censored."""
        return self._param_df["censored"].to_numpy(dtype=bool)

    @property
    def guesses(self) -> np.ndarray:
        """np.ndarray: The initial guess for each parameter."""
        return self._param_df["guess"].to_numpy()

    @property
    def guesses_transformed(self) -> np.ndarray:
        """np.ndarray: The initial guesses in the transformed space."""
        return self.transform(self.guesses)
    
    @property
    def is_fixed(self) -> np.ndarray:
        """np.ndarray: The initial guess for each parameter."""
        return self._param_df["fixed"].to_numpy(dtype=bool)
