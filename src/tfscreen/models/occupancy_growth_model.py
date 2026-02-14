
import numpy as np

class OccupancyGrowthModel:
    """
    A unified model relating transcription factor occupancy (theta) to bacterial growth,
    accounting for dilution and physiological transitions (Baryani shift).

    The model describes growth in two phases:
    1. Pre-growth (t < 0): Growth at rate mu1 (determined by condition_pre).
    2. Selection (t > 0): Dilution occurs at t=0, followed by growth transitioning
       from mu1 to mu2 (determined by condition_sel) with a shift delay (tau)
       and sharpness (k_sharp).

    Mathematical Formulation:
    ln_cfu(t) = ln_cfu0 + integral(mu(t') dt' from -t_pre to t) - ln(dilution_factor * step(t))

    where mu(t) transitions from mu1 to mu2.
    """

    def __init__(self):
        pass

    @staticmethod
    def calc_baryani_integral(t, tau, k):
        """
        Calculate the integral of the sigmoid transition function.
        
        The interaction is defined as:
        I(t) = integral_0^t [1 / (1 + exp(-k(t' - tau)))] dt'
             = (1/k) * ln( (1 + exp(k(t - tau))) / (1 + exp(-k*tau)) )
        
        However, to represent the integral of the transition from 0 to 1, we use:
        Sigmoid S(t) goes from 0 to 1.
        Growth rate mu(t) = mu1 + (mu2 - mu1) * S(t)
        Integral = mu1*t + (mu2 - mu1) * Integral_S(t)

        Standard Baryani shift usually implies a lag phase where growth is 0 then
        transitions to mu_max. Here we transition from mu1 to mu2.

        This function computes the integral of the sigmoid function:
        S(t') = 1 / (1 + exp(-k*(t' - tau)))
        
        Using logaddexp for numerical stability:
        Integral = (1/k) * ( logaddexp(0, k*(t - tau)) - logaddexp(0, -k*tau) )

        Parameters
        ----------
        t : array-like
            Time points (must be >= 0).
        tau : float or array-like
            Time of half-maximal transition (lag time).
        k : float or array-like
            Steepness of the transition.

        Returns
        -------
        array-like
            Value of the integral at time t.
        """
        # Ensure k is not too small to avoid division by zero
        k_eff = np.maximum(k, 1e-6)
        
        term1 = np.logaddexp(0, k_eff * (t - tau))
        term0 = np.logaddexp(0, -k_eff * tau)
        
        return (term1 - term0) / k_eff

    def predict_trajectory(self,
                           t_pre,
                           t_sel,
                           ln_cfu0,
                           mu1,
                           mu2,
                           dilution,
                           tau=0.0,
                           k_sharp=1.0):
        """
        Predict ln_cfu for a full experimental trajectory including pre-growth and selection.

        Parameters
        ----------
        t_pre : float or array-like
            Duration of pre-growth phase (positive).
        t_sel : float or array-like
            Time in selection phase. Can be negative (implying pre-growth phase).
            If t_sel < 0, it represents time -t_pre relative to dilution.
            However, usually t_sel is the observed time.
            The model assumes:
            - Start at t = -t_pre with ln_cfu0.
            - Grow at mu1 until t = 0.
            - Dilute at t = 0 (instantaneous drop).
            - Grow transitioning from mu1 to mu2 for t > 0.
        ln_cfu0 : float or array-like
            Initial ln_cfu at start of pre-growth (t = -t_pre).
        mu1 : float or array-like
            Growth rate in pre-growth phase.
        mu2 : float or array-like
            Growth rate in selection phase (asymptotic).
        dilution : float
            Dilution factor (e.g. 0.01 for 1:100 dilution).
            The population drops by ln(dilution) at t=0.
        tau : float or array-like, optional
            Lag time / transition time for Baryani shift, by default 0.0.
        k_sharp : float or array-like, optional
            Steepness of transition, by default 1.0.

        Returns
        -------
        array-like
            Predicted ln_cfu values corresponding to t_sel.
        """
        
        # Ensure inputs are numpy arrays for broadcasting
        t_pre = np.asarray(t_pre)
        t_sel = np.asarray(t_sel)
        ln_cfu0 = np.asarray(ln_cfu0)
        mu1 = np.asarray(mu1)
        mu2 = np.asarray(mu2)
        tau = np.asarray(tau)
        k_sharp = np.asarray(k_sharp)
        
        # Initialize output with same shape as broadcasted inputs
        # We broadcast everything against t_sel first
        shape = np.broadcast_shapes(t_sel.shape, t_pre.shape, ln_cfu0.shape, mu1.shape, mu2.shape)
        
        # Phase 1: Pre-growth (t_sel < 0)
        # Growth is linear from ln_cfu0 with slope mu1.
        # The time elapsed since -t_pre is (t_pre + t_sel).
        # ln_cfu = ln_cfu0 + mu1 * (t_pre + t_sel)
        pred_pre = ln_cfu0 + mu1 * (t_pre + t_sel)
        
        # Phase 2: Selection (t_sel >= 0)
        # State after dilution (t = 0+):
        ln_cfu_start_sel = (ln_cfu0 + mu1 * t_pre) + np.log(dilution)
        
        # Growth during selection:
        # Integral part for Baryani shift
        # mu(t) = mu1 + (mu2 - mu1) * S(t)
        # integral mu(t) = mu1 * t + (mu2 - mu1) * Integral_S(t)
        
        # Use maximum(0, t_sel) to avoid issues if t_sel < 0 passed to integral
        t_sel_pos = np.maximum(0.0, t_sel)
        
        integral_term = self.calc_baryani_integral(t_sel_pos, tau, k_sharp)
        
        growth_sel = mu1 * t_sel_pos + (mu2 - mu1) * integral_term
        
        pred_post = ln_cfu_start_sel + growth_sel
        
        # Combine
        # If t_sel < 0, use pred_pre. Else use pred_post.
        return np.where(t_sel < 0, pred_pre, pred_post)

