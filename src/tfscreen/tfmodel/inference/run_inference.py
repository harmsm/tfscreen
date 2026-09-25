
import jax
from jax import random
from jax import numpy as jnp

import numpyro
from numpyro.handlers import seed, trace

from numpyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
)
from numpyro.infer import autoguide
from numpyro.infer.initialization import (
    init_to_median,
    init_to_uniform,
)
from numpyro.distributions.transforms import IdentityTransform
from numpyro.optim import ClippedAdam
import numpy as np
import dill
from functools import partial
from tqdm.auto import tqdm

from tfscreen.tfmodel.inference.batch_safety import (
    find_orchestrator_batch_dependent_latents,
    orchestrator_latent_dimension,
)
from tfscreen.tfmodel.inference import convergence as conv
from tfscreen.tfmodel.inference.initialization import (
    AUTO_LOC_SUFFIX,
    component_guide_init,
    component_guide_map,
    site_prior_sds,
    site_values,
    trace_model_sites,
)

# The convergence window is split into this many blocks for the parameter
# test (each block's parameter mean is accumulated on the device), and the
# loop hands control back to the host once per block.
PARAM_BLOCKS = 8

# Every window spans at least this many epochs, so with mini-batching each
# genotype is visited several times per window.
MIN_WINDOW_EPOCHS = 10

# Parameters never tracked for movement: dense covariance factors (their
# diagonal is tracked through the derived posterior SD instead).
_UNTRACKED_PARAM_TAGS = ("scale_tril", "cov_factor")

# Location/scale suffix pairs, checked in order (``_auto_loc`` also ends in
# ``_loc``).
_LOC_SCALE_SUFFIXES = (("_auto_loc", "_auto_scale"),
                       ("_locs", "_scales"),
                       ("_loc", "_scale"))

# Autoguides selectable through setup_svi(guide_type=...), keyed by the
# snake_case form of the numpyro class name ('delta' keeps its historical short
# name).  'component' -- the guide assembled from the model components -- is
# handled separately.
AUTOGUIDES = {
    "delta": autoguide.AutoDelta,
    "auto_normal": autoguide.AutoNormal,
    "auto_diagonal_normal": autoguide.AutoDiagonalNormal,
    "auto_multivariate_normal": autoguide.AutoMultivariateNormal,
    "auto_low_rank_multivariate_normal": autoguide.AutoLowRankMultivariateNormal,
}
GUIDE_TYPES = ("component",) + tuple(AUTOGUIDES)

# guide_kwargs each guide accepts (init_loc_fn is set through init_values).
_AUTOGUIDE_KWARGS = {
    "component": set(),
    "delta": set(),
    "auto_normal": {"init_scale"},
    "auto_diagonal_normal": {"init_scale"},
    "auto_multivariate_normal": {"init_scale"},
    "auto_low_rank_multivariate_normal": {"init_scale", "rank"},
}

# Case-insensitive aliases: numpyro class names, plus 'auto_delta'.
_GUIDE_ALIASES = {cls.__name__.lower(): name for name, cls in AUTOGUIDES.items()}
_GUIDE_ALIASES["auto_delta"] = "delta"

# Warn when the dense auto_multivariate_normal covariance exceeds this (GB).
_DENSE_GUIDE_WARN_GB = 4.0


def _init_to_value_or(site=None, values=None, fallback=init_to_uniform):
    """
    ``init_to_value`` with a choice of strategy for sites missing from
    ``values`` (numpyro's always falls back to ``init_to_uniform``).
    """
    if site is None:
        return partial(_init_to_value_or, values=values, fallback=fallback)
    if site["type"] == "sample" and not site["is_observed"] \
            and site["name"] in (values or {}):
        return values[site["name"]]
    return fallback(site)


def resolve_guide_type(guide_type):
    """Return the canonical ``GUIDE_TYPES`` name for ``guide_type``."""

    key = str(guide_type).lower()
    key = _GUIDE_ALIASES.get(key, key)
    if key not in GUIDE_TYPES:
        raise ValueError(
            f"guide_type '{guide_type}' not recognized. It should be one of "
            f"{list(GUIDE_TYPES)} (numpyro class names such as 'AutoNormal' "
            f"are also accepted)."
        )
    return key


def check_guide_kwargs(guide_type, guide_kwargs):
    """Raise ValueError if ``guide_kwargs`` has keys ``guide_type`` rejects."""

    guide_type = resolve_guide_type(guide_type)
    accepted = _AUTOGUIDE_KWARGS[guide_type]
    unknown = set(guide_kwargs or {}) - accepted
    if unknown:
        raise ValueError(
            f"guide option(s) {sorted(unknown)} not accepted by guide_type "
            f"'{guide_type}' (accepted: {sorted(accepted) or 'none'})."
        )

_HDF5_MAX_CHUNK_BYTES = 1 << 30  # 1 GiB; HDF5 hard-limit is 4 GiB

def _safe_chunks(first_dim, trailing_shape, dtype):
    """Return a chunk tuple whose total byte size stays under _HDF5_MAX_CHUNK_BYTES."""
    trailing = int(np.prod(trailing_shape)) if trailing_shape else 1
    item = np.dtype(dtype).itemsize
    safe_first = max(1, _HDF5_MAX_CHUNK_BYTES // (trailing * item))
    return (min(first_dim, safe_first),) + trailing_shape

import os
import warnings
import h5py
class RunInference:
    """
    Manages the SVI (Stochastic Variational Inference) process for a model.
    
    This class handles SVI setup, optimization loops, checkpointing, 
    convergence checking, and posterior sample generation. It is designed
    to interface with a 'model' object that defines the JAX/Numpyro model,
    data, and initial parameters.
    """

    def __init__(self,model,seed):
        """
        Initialize the RunInference class.

        Parameters
        ----------
        model : object
            A model object that must expose the following attributes:
            - `data` (flax.struct.dataclass): Data object, expected to have `num_genotype`.
            - `priors` (flax.struct.dataclass): Data object holding model priors
            - `jax_model` (callable): The Numpyro model.
            - `jax_model_guide` (callable): The guide for the Numpyro model.
        seed : int
            Random seed for JAX PRNG key generation.
        """
        
        required_attr = ["data",
                         "priors",
                         "jax_model",
                         "jax_model_guide"]
        for attr in required_attr:
            if not hasattr(model,attr):
                raise ValueError(f"`model` must have attribute {attr}")
        
        self.model = model
        self._seed = seed
        self._main_key = random.PRNGKey(self._seed)
        self._current_step = 0

        # Step size and gradient clip of the optimizer built by setup_svi.
        # run_optimization rebuilds the optimizer when it cuts the step size.
        self._step_size = None
        self._adam_clip_norm = None

        # Convergence-monitor state: restored from a checkpoint (resume) and
        # written back into checkpoints; the monitor itself lives in
        # run_optimization.
        self._monitor_state = None
        self._monitor = None

        # Calculate iterations per epoch
        num_genotypes = self.model.data.num_genotype

        # Determine batch size by dry-running get_random_idx
        init_batch_key = int(self.get_key()[1])
        test_idx = self.model.get_random_idx(init_batch_key,num_batches=1)
        batch_size = len(test_idx.flatten())

        self._iterations_per_epoch = int(np.ceil(num_genotypes / batch_size))
        self._batch_size = batch_size

        # Set by setup_svi and written into checkpoints.
        self._guide_type = None
        self._guide_kwargs = {}


    def setup_svi(self,
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  elbo_num_particles=2,
                  guide_type="delta",
                  guide_kwargs=None,
                  init_values=None):
        """
        Set up SVI.

        Parameters
        ----------
        adam_step_size : float or callable, optional
            Step size for the ClippedAdam optimizer. Can be a fixed float or
             a callable (e.g., an optax schedule).
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.
        guide_type : str, optional
            Variational family (default 'delta'); one of ``GUIDE_TYPES``.

            - 'component': the guide assembled from the model components.
            - 'delta': numpyro ``AutoDelta`` (MAP estimation).
            - 'auto_normal', 'auto_diagonal_normal',
              'auto_multivariate_normal', 'auto_low_rank_multivariate_normal':
              the numpyro autoguide of the same name.

            numpyro class names (e.g. 'AutoNormal') are also accepted,
            case-insensitively.
        guide_kwargs : dict or None, optional
            Extra keyword arguments for an autoguide: ``init_scale`` (every
            autoguide except 'delta') and ``rank``
            ('auto_low_rank_multivariate_normal' only).
        init_values : dict or None, optional
            Constrained values keyed by model site name, used to initialize an
            autoguide's location.  Sites not present fall
            back to numpyro's uniform initialization.  Autoguides only.

        Returns
        -------
        numpyro.infer.SVI
            An SVI object

        Raises
        ------
        ValueError
            If ``guide_type`` is unknown, or if ``guide_kwargs``/``init_values``
            are given for a guide that does not take them.  Whether the model
            can be fit with an autoguide at all is checked when fitting starts
            (``run_optimization``).
        """

        guide_type = resolve_guide_type(guide_type)
        guide_kwargs = dict(guide_kwargs or {})
        check_guide_kwargs(guide_type, guide_kwargs)

        if guide_type == "component":
            if init_values is not None:
                raise ValueError(
                    "init_values applies only to autoguides, not "
                    "guide_type='component'."
                )
            guide = self.model.jax_model_guide
        else:
            ctor_kwargs = dict(guide_kwargs)
            if init_values is not None:
                # Sites without a value fall back to the autoguide's own
                # default initialization (median for AutoDelta, uniform for
                # the others).
                fallback = (init_to_median if guide_type == "delta"
                            else init_to_uniform)
                ctor_kwargs["init_loc_fn"] = partial(_init_to_value_or,
                                                     values=init_values,
                                                     fallback=fallback)
            guide = AUTOGUIDES[guide_type](self.model.jax_model, **ctor_kwargs)

        # Recorded in checkpoints so the same guide can be rebuilt on restore.
        self._guide_type = guide_type
        self._guide_kwargs = guide_kwargs

        self._step_size = adam_step_size
        self._adam_clip_norm = adam_clip_norm
        optimizer = ClippedAdam(step_size=adam_step_size,
                                clip_norm=adam_clip_norm)
        
        svi = SVI(self.model.jax_model,
                  guide,
                  optimizer,
                  loss=Trace_ELBO(num_particles=elbo_num_particles))

        return svi

    def _check_autoguide(self, guide):
        """
        Refuse an autoguide the model cannot support; report dense-guide size.

        Autoguides size their parameters from a traced batch, so a latent whose
        shape follows the genotype batch would get one parameter per batch
        position instead of per genotype -- at any batch size, because the
        full-batch index is reshuffled every step.  The component guide is
        exempt: it holds library-sized parameters and slices them itself.
        Called when fitting starts, since fitting is what creates the aliasing.
        """

        guide_type = self._guide_type or type(guide).__name__

        found = find_orchestrator_batch_dependent_latents(self.model)
        if found:
            raise ValueError(
                f"guide_type '{guide_type}' cannot fit latent site(s) "
                f"{sorted(found)}: their shape follows the genotype mini-batch "
                f"rather than the library, so an autoguide would assign one "
                f"genotype's value to another. Use guide_type='component', or "
                f"a component whose latents are library-sized (e.g. "
                f"theta_growth_noise='logit_normal' instead of 'beta'). See "
                f"inference/batch_safety.py."
            )

        if isinstance(guide, autoguide.AutoMultivariateNormal):
            dim = orchestrator_latent_dimension(self.model)
            # scale_tril plus Adam's two moment estimates, float32.
            gigabytes = 3 * dim * dim * 4 / 1e9
            msg = (f"auto_multivariate_normal: {dim} latent dimensions; the "
                   f"dense covariance needs ~{gigabytes:.3g} GB (parameters "
                   f"plus optimizer state).")
            if gigabytes > _DENSE_GUIDE_WARN_GB:
                warnings.warn(msg + " Consider "
                              "'auto_low_rank_multivariate_normal'.")
            else:
                print(msg, flush=True)

    def run_optimization(self,
                         svi,
                         svi_state=None,
                         init_params=None,
                         out_prefix="tfs",
                         convergence_window_steps=2000,
                         patience=3,
                         convergence_z=3.0,
                         loss_rtol=1e-6,
                         param_tolerance=0.05,
                         final_step_size=None,
                         step_size_cut=0.1,
                         checkpoint_interval=10,
                         max_num_epochs=10000000,
                         init_param_jitter=0.1,
                         epoch_checkpoint_interval=1000):
        """
        Run the optimization loop until convergence or ``max_num_epochs``.

        The loop runs in tumbling windows of optimizer steps.  At the end of
        each window ``inference/convergence.py`` tests whether the loss is
        still improving (relative to its own noise) and whether any parameter
        is still moving (relative to its posterior or prior width).  After
        ``patience`` consecutive windows without loss improvement the step
        size is cut by ``step_size_cut``; once it has reached
        ``final_step_size``, ``patience`` consecutive windows with neither
        loss improvement nor parameter movement are convergence.  Every window
        is recorded in ``{out_prefix}_convergence.csv``.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            The SVI object from `setup_svi`.  Its optimizer is replaced (same
            state, smaller step size) at each step-size cut.
        svi_state : Any, optional
            An existing SVI state to continue from, or the path of a checkpoint
            to resume (restoring the step count, step size and convergence
            stage).  If None, a new state is initialized and the step count
            starts at zero.
        init_params : dict, optional
            Initial parameters.
        out_prefix : str, optional
            Root name for output files (checkpoints, losses).
        convergence_window_steps : int, optional
            Length of a convergence window in optimizer steps (default 2000).
            Raised to at least ``MIN_WINDOW_EPOCHS`` epochs and rounded up to a
            multiple of ``PARAM_BLOCKS``.
        patience : int, optional
            Consecutive windows without loss improvement required for a
            step-size cut, and (at ``final_step_size``) without loss
            improvement or parameter movement required for convergence
            (default 3).
        convergence_z : float, optional
            Standard errors of change attributed to noise (default 3).
        loss_rtol : float, optional
            Loss changes smaller than this fraction of the loss per window
            count as a plateau even when significant (default 1e-6); matters
            only for (near-)deterministic losses.
        param_tolerance : float, optional
            Parameter movement per window, beyond noise, that still counts as
            a plateau (default 0.05), in posterior SDs (guide scale), prior SDs
            (MAP locations), or log/logit units (positive or bounded
            parameters).
        final_step_size : float or None, optional
            Smallest step size.  None (default) or a value >= the current step
            size disables cuts: the first sustained plateau is convergence.
        step_size_cut : float, optional
            Factor applied to the step size at each cut (default 0.1).
        checkpoint_interval : int, optional
            Frequency (in epochs) to write checkpoints.
        max_num_epochs : int, optional
            Maximum number of optimization epochs to run.
        init_param_jitter : float, optional
            amount of jitter to add to init_params. To turn off, set to 0.
        epoch_checkpoint_interval : int or None, optional
            Frequency (in epochs) to write numbered epoch checkpoints to a
            ``checkpoints/`` subdirectory alongside ``out_prefix``. Files are
            named ``{epoch:07d}_checkpoint.pkl``. Set to None or 0 to
            disable (default 1000).

        Returns
        -------
        svi_state : Any
            The final SVI state.
        params : dict
            The final optimized parameters.
        converged : bool
            True if the run stopped because it converged (see above).

        Raises
        ------
        RuntimeError
            If parameters explode to NaN during optimization.
        FileExistsError
            If a numbered epoch checkpoint file already exists.
        ValueError
            If ``svi`` uses a numpyro autoguide (AutoDelta included) and the
            model has latents whose shape follows the genotype mini-batch
            (see ``inference/batch_safety.py``), or if step-size cuts are
            requested for an optimizer built with a step-size schedule.
        """

        # Refuse an autoguide the model cannot support before any fitting.
        if isinstance(svi.guide, autoguide.AutoGuide):
            self._check_autoguide(svi.guide)

        # Add jitter to the input parameters if they are specified
        if init_params is not None:
            init_params = self._jitter_init_parameters(init_params=init_params,
                                                       init_param_jitter=init_param_jitter)

        # Put the data on to the gpu
        data_on_gpu = jax.device_put(self.model.data)

        # Create an initial batch to initialize SVI
        gpu_batch_idx = jax.device_put(self.model.get_random_idx())
        batch_data = self.model.get_batch(data_on_gpu, gpu_batch_idx)

        # Initialize svi with a batch of data
        init_key = self.get_key()
        initial_svi_state = svi.init(init_key,
                                     init_params=init_params,
                                     priors=self.model.priors,
                                     data=batch_data)

        self._monitor_state = None
        if svi_state is None:
            svi_state = initial_svi_state
            self._current_step = 0
        elif isinstance(svi_state,str):
            if os.path.isfile(svi_state):
                svi_state = self._restore_checkpoint(svi_state)
            else:
                raise ValueError(
                    f"svi_state '{svi_state}' is not valid"
                )

        # --- Step size and convergence monitor ---

        step_size = self._step_size
        if step_size is None:
            step_size = getattr(svi.optim, "step_size", None)
        if callable(step_size):
            if final_step_size is not None:
                raise ValueError(
                    "step-size cuts (final_step_size) need a constant step "
                    "size from setup_svi, not a schedule."
                )
            monitor_step_size = np.nan
        else:
            monitor_step_size = float(step_size) if step_size is not None else np.nan

        monitor = conv.ConvergenceMonitor(
            step_size=monitor_step_size,
            final_step_size=final_step_size,
            step_size_cut=step_size_cut,
            patience=patience,
            z=convergence_z,
            loss_rtol=loss_rtol,
            param_tolerance=param_tolerance,
        )
        if self._monitor_state is not None:
            monitor.load_state_dict(self._monitor_state)
            if monitor.step_size != monitor_step_size:
                print(f"Resuming at the checkpoint's step size "
                      f"{monitor.step_size:.3g}.", flush=True)
                self._set_step_size(svi, monitor.step_size)
        self._monitor = monitor

        # --- Window geometry ---

        ipe = self._iterations_per_epoch
        window_steps = max(int(convergence_window_steps),
                           MIN_WINDOW_EPOCHS * ipe, 3 * PARAM_BLOCKS)
        block_steps = int(np.ceil(window_steps / PARAM_BLOCKS))
        window_steps = block_steps * PARAM_BLOCKS

        # --- Parameters tracked for movement ---

        unconstrained = svi.optim.get_params(svi_state.optim_state)
        specs = self._param_normalizer_specs(svi, svi_state, unconstrained)
        zero_acc = {k: jnp.zeros_like(unconstrained[k]) for k in specs}

        # JAX-optimized update function for use with lax.scan.  Each step also
        # adds the tracked (unconstrained) parameters to an accumulator, so
        # the host gets per-block parameter means without per-step transfers.
        def scan_fn(data_on_gpu, carry, indices):
            state, acc = carry
            batch = self.model.get_batch(data_on_gpu, indices)
            new_state, loss = svi.update(state,
                                         priors=self.model.priors,
                                         data=batch)
            current = svi.optim.get_params(new_state.optim_state)
            acc = {k: acc[k] + current[k] for k in acc}
            return (new_state, acc), loss

        # Built as a fresh closure so a step-size cut (which swaps svi.optim)
        # re-traces rather than hitting a cached compilation. The data are a
        # formal argument to avoid capturing them as a constant (expensive
        # constant folding for large datasets).
        def build_scan():
            return jax.jit(lambda data, state, acc, indices: jax.lax.scan(
                partial(scan_fn, data), (state, acc), indices))

        fast_scan = build_scan()

        # Initialize loss and convergence files
        self._write_losses(np.array([]), out_prefix)
        self._write_convergence(None, out_prefix)

        converged = False
        total_steps = max_num_epochs * ipe

        checkpoint_interval_steps = checkpoint_interval * ipe

        # Track next checkpoint in steps
        # If resuming, we want to write checkpoint at the next multiple of
        # checkpoint_interval_steps.
        self._next_checkpoint_step = ((self._current_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps

        # Set up epoch checkpoint tracking
        if epoch_checkpoint_interval:
            epoch_checkpoint_interval_steps = epoch_checkpoint_interval * ipe
            self._next_epoch_checkpoint_step = (
                (self._current_step // epoch_checkpoint_interval_steps) + 1
            ) * epoch_checkpoint_interval_steps
            out_dir = os.path.dirname(out_prefix)
            self._epoch_checkpoints_dir = os.path.join(out_dir, "checkpoints") if out_dir else "checkpoints"
            os.makedirs(self._epoch_checkpoints_dir, exist_ok=True)
        else:
            epoch_checkpoint_interval_steps = None

        print(f"Convergence window: {window_steps} steps "
              f"({window_steps / ipe:.4g} epochs); patience {patience} "
              f"windows; step size {monitor.step_size:.3g}"
              + (f" -> {monitor.final_step_size:.3g}"
                 if not monitor.at_floor else "")
              + f"; tracking {len(specs)} parameter arrays.", flush=True)

        # Current window
        window_losses = []
        block_means = []
        block_centers = []
        window_filled = 0

        current_optimization_step = 0
        while current_optimization_step < total_steps:

            # Determine size of this block
            block_size = min(block_steps, total_steps - current_optimization_step)

            # Generate a block of random indices (using NumPy/Python)
            block_idx = self.model.get_random_idx(num_batches=block_size)
            if block_idx.ndim == 1:
                block_idx = block_idx.reshape(1,-1)
            gpu_block_idx = jax.device_put(block_idx)

            # Run the block of updates using lax.scan (entirely on GPU)
            (svi_state, acc), block_losses = fast_scan(data_on_gpu, svi_state,
                                                       zero_acc, gpu_block_idx)

            # Convert JAX array to NumPy for host-side metadata management
            # Ensure it is at least 1D for IO
            interval_losses = np.atleast_1d(np.array(block_losses))

            # Update counters
            current_optimization_step += block_size
            self._current_step += block_size

            # Accumulate the window
            window_losses.append(interval_losses)
            block_means.append({k: v / block_size for k, v in acc.items()})
            block_centers.append(window_filled + block_size / 2)
            window_filled += block_size

            # stdout
            print(f"Step: {self._current_step:10d}, "
                  f"Loss: {np.median(interval_losses):10.5e}, "
                  f"Step size: {monitor.step_size:9.3e}, "
                  f"Plateau: {monitor.plateau_count}/{patience}", flush=True)

            # Check for explosion in parameters
            params = svi.get_params(svi_state)
            for k in params:
                if np.any(np.isnan(params[k])):

                    nan_params = [(k,params[k]) for k in params]
                    raise RuntimeError(
                        f"model exploded (observed at step {self._current_step}). "
                        f"NaN params: {nan_params}."
                    )

            # Write outputs (checkpoints and losses)
            self._write_losses(interval_losses, out_prefix)

            # End of a convergence window
            decision = None
            if window_filled >= window_steps:
                decision = self._end_window(svi, svi_state, monitor, specs,
                                            np.concatenate(window_losses),
                                            block_means, block_centers,
                                            window_filled)
                self._write_convergence(monitor.last, out_prefix)
                print("Convergence window, " + monitor.describe(), flush=True)
                window_losses = []
                block_means = []
                block_centers = []
                window_filled = 0

                if decision == conv.CUT:
                    self._set_step_size(svi, monitor.step_size)
                    fast_scan = build_scan()

            # Check if we should write a checkpoint
            if self._current_step >= self._next_checkpoint_step:
                self._write_checkpoint(svi_state, out_prefix)
                self._next_checkpoint_step += checkpoint_interval_steps

            # Check if we should write a numbered epoch checkpoint
            if epoch_checkpoint_interval_steps is not None:
                if self._current_step >= self._next_epoch_checkpoint_step:
                    current_epoch = self._current_step // ipe
                    self._write_epoch_checkpoint(svi_state, current_epoch)
                    self._next_epoch_checkpoint_step += epoch_checkpoint_interval_steps

            if decision == conv.CONVERGED:
                converged = True
                # Final checkpoint before exiting
                self._write_checkpoint(svi_state, out_prefix)
                break

        # Write a final checkpoint when the loop exits by reaching max_num_epochs
        # (convergence already writes its own checkpoint via the break path above).
        if not converged and total_steps > 0:
            self._write_checkpoint(svi_state, out_prefix)

        if total_steps > 0:
            if converged:
                print(f"Converged at step {self._current_step} (step size "
                      f"{monitor.step_size:.3g}, {monitor.num_cuts} cut(s)): "
                      f"no significant loss improvement or parameter movement "
                      f"for {patience} windows.", flush=True)
            else:
                print(f"Stopped at step {self._current_step} "
                      f"(max_num_epochs) without converging. Last window: "
                      f"{monitor.describe()}", flush=True)

        # Get final parameters
        params = svi.get_params(svi_state)

        return svi_state, params, converged

    def _set_step_size(self, svi, step_size):
        """
        Give ``svi`` a ClippedAdam with a new constant step size.

        The optimizer state (step count, parameters, Adam moments) does not
        depend on the step size, so the existing state carries over.
        """

        clip_norm = self._adam_clip_norm
        if clip_norm is None:
            clip_norm = getattr(svi.optim, "clip_norm", 10.0)
        svi.optim = ClippedAdam(step_size=step_size, clip_norm=clip_norm)
        self._step_size = step_size

    @staticmethod
    def _param_transforms(svi):
        """``{param name: transform}`` recorded by ``svi.init``, if exposed."""
        args = getattr(getattr(svi, "constrain_fn", None), "args", None)
        if args and isinstance(args[0], dict):
            return args[0]
        return {}

    def _param_normalizer_specs(self, svi, svi_state, unconstrained):
        """
        How to normalize each tracked parameter's movement.

        Returns ``{param name: spec}`` where spec is one of

        - ``("unit",)``: movement taken as is -- parameters on a positive or
          bounded support (tracked in log/logit units, so relative), and real
          parameters with no better reference;
        - ``("scale", scale_name)``: a guide location divided by its paired
          guide scale (posterior SD);
        - ``("auto_continuous",)``: an AutoContinuous ``auto_loc`` divided by
          the posterior SD derived from its scale parameters;
        - ``("prior", sd)``: a MAP (AutoDelta) location divided by the prior
          SD of its site.
        """

        transforms = self._param_transforms(svi)
        shapes = {k: jnp.shape(v) for k, v in unconstrained.items()}

        prior_sds = {}
        if self._guide_type == "delta":
            prior_sds = self._delta_prior_sds(svi, svi_state)

        specs = {}
        for name in unconstrained:
            if any(tag in name for tag in _UNTRACKED_PARAM_TAGS):
                continue
            transform = transforms.get(name)
            if transform is not None and not isinstance(transform,
                                                        IdentityTransform):
                specs[name] = ("unit",)
                continue

            if name == "auto_loc":
                specs[name] = ("auto_continuous",)
                continue

            partner = None
            for loc_suffix, scale_suffix in _LOC_SCALE_SUFFIXES:
                if name.endswith(loc_suffix):
                    candidate = name[:-len(loc_suffix)] + scale_suffix
                    if shapes.get(candidate) == shapes[name]:
                        partner = candidate
                    break

            site = (name[:-len(AUTO_LOC_SUFFIX)]
                    if name.endswith(AUTO_LOC_SUFFIX) else None)
            if partner is not None:
                specs[name] = ("scale", partner)
            elif site is not None and site in prior_sds \
                    and jnp.shape(prior_sds[site]) == shapes[name]:
                specs[name] = ("prior", prior_sds[site])
            else:
                specs[name] = ("unit",)
        return specs

    def _delta_prior_sds(self, svi, svi_state):
        """Prior SD of each site at the current MAP point ({} on failure)."""
        try:
            constrained = svi.get_params(svi_state)
            substitutions = {k[:-len(AUTO_LOC_SUFFIX)]: v
                             for k, v in constrained.items()
                             if k.endswith(AUTO_LOC_SUFFIX)}
            sites = trace_model_sites(self.model.jax_model,
                                      self.model.priors,
                                      self._trace_batch(),
                                      substitutions=substitutions)
            return site_prior_sds(sites)
        except Exception as err:  # pragma: no cover - defensive
            print(f"Could not compute prior SDs for MAP parameter movement "
                  f"({err}); movement is measured in unconstrained units.",
                  flush=True)
            return {}

    def _trace_batch(self):
        """
        A deterministic batch for tracing: the first ``batch_size`` genotypes
        of the full binding-first index (latent sites are library-sized
        whatever the batch).
        """
        data = jax.device_put(self.model.data)
        idx = jnp.asarray(self.model.data.batch_idx)[:self._batch_size]
        return self.model.get_batch(data, idx)

    def site_values(self, values):
        """
        Constrained site values found in ``values`` (guesses, a MAP result or
        both), keyed by model site name.  See ``inference/initialization.py``.
        """
        batch = self._trace_batch()
        sites = trace_model_sites(self.model.jax_model, self.model.priors,
                                  batch)
        guide_map, _, _ = component_guide_map(self.model.jax_model_guide,
                                              self.model.priors, batch)
        return site_values(values, sites, guide_map)

    def component_guide_start(self, values, guesses=None, init_scale=None):
        """
        Initial component-guide parameters starting at the given site values.

        Parameters
        ----------
        values : dict
            Constrained site values (``site_values``).
        guesses : dict or None, optional
            Configured guesses.  Those keyed by component-guide param name are
            kept (and overridden where ``values`` gives the same site).
        init_scale : float or None, optional
            Upper bound on every mapped guide scale at the start.

        Returns
        -------
        dict
            ``init_params`` for ``run_optimization`` with the component guide.
        """

        guesses = dict(guesses or {})
        guide_map, unmatched, defaults = component_guide_map(
            self.model.jax_model_guide, self.model.priors, self._trace_batch())

        init_params = {k: v for k, v in guesses.items() if k in defaults}
        start_defaults = dict(defaults)
        start_defaults.update(init_params)

        translated, skipped = component_guide_init(values, guide_map,
                                                   start_defaults,
                                                   init_scale=init_scale)
        init_params.update(translated)

        n_loc = sum(1 for e in guide_map.values() if e["loc"] in translated)
        print(f"Guide start: {n_loc} of {len(guide_map) + len(unmatched)} "
              f"guide sites start at the given values"
              + (f"; scales capped at {init_scale:g}"
                 if init_scale is not None else "")
              + ".", flush=True)
        if unmatched:
            print(f"  guide sites without a recognized location parameter "
                  f"(left at their defaults): {unmatched}", flush=True)
        if skipped:
            print(f"  non-positive values for LogNormal-guided sites (left "
                  f"at their defaults): {skipped}", flush=True)
        return init_params

    @staticmethod
    def _normalizer(spec, constrained):
        """Evaluate one normalizer spec at the current constrained params."""
        kind = spec[0]
        if kind == "scale":
            norm = constrained[spec[1]]
        elif kind == "auto_continuous":
            if "auto_scale_tril" in constrained:
                tril = constrained["auto_scale_tril"]
                norm = jnp.sqrt(jnp.sum(tril * tril, axis=-1))
            elif "auto_cov_factor" in constrained:
                factor = constrained["auto_cov_factor"]
                norm = jnp.sqrt(jnp.sum(factor * factor, axis=-1)
                                + constrained["auto_scale"] ** 2)
            elif "auto_scale" in constrained:
                norm = constrained["auto_scale"]
            else:
                norm = 1.0
        elif kind == "prior":
            norm = spec[1]
        else:
            norm = 1.0
        return jnp.maximum(jnp.asarray(norm, dtype=float), 1e-12)

    def _end_window(self, svi, svi_state, monitor, specs, losses,
                    block_means, block_centers, window_steps):
        """Run both convergence tests on a finished window; return the decision."""

        loss_stats = conv.loss_trend(losses)

        constrained = svi.get_params(svi_state)
        centers = np.asarray(block_centers, dtype=float)
        floor = monitor.drift_floor(window_steps)
        excess_summary = {}
        drift_summary = {}
        for name, spec in specs.items():
            stacked = jnp.stack([b[name] for b in block_means])
            drift, excess = conv.param_drift(stacked, centers, window_steps,
                                             self._normalizer(spec, constrained),
                                             monitor.z, floor=floor)
            excess_summary[name] = conv.summarize_excess(excess)
            drift_summary[name] = conv.summarize_excess(jnp.abs(drift))

        return monitor.end_window(self._current_step, loss_stats,
                                  excess_summary, drift_summary)

    def _get_genotype_dim_map(self):
        """
        Identify which sites are in the genotype plate and what that dimension is.

        Returns
        -------
        dict
            Dictionary mapping site names to their genotype dimension index.
        """

        # Use a minimal probe batch so that data.batch_size matches the actual
        # tensor dimensions when tracing.  MAP checkpoints trained with
        # mini-batching store data.binding.batch_size == mini-batch size while
        # the full tensors have num_genotype entries; tracing with raw
        # self.model.data causes a plate-size / tensor-size mismatch.
        data_on_gpu = jax.device_put(self.model.data)
        probe_size = min(2, self.model.data.num_genotype)
        probe_idx = jnp.arange(probe_size)
        probe_data = self.model.get_batch(data_on_gpu, probe_idx)

        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        model_trace = traced_model.get_trace(data=probe_data,
                                             priors=self.model.priors)

        dim_map = {}
        genotype_dim = -1 # default fallback

        # First pass: find a site with the genotype plate to identify the dim index
        for name, site in model_trace.items():
            for frame in site.get("cond_indep_stack", []):
                if "genotype" in frame.name.lower():
                    genotype_dim = frame.dim
                    break
            if genotype_dim != -1:
                break

        # Second pass: map all sites that are in the plate or match the genotype size
        for name, site in model_trace.items():
            if site["type"] not in ["sample", "deterministic"]:
                continue

            # Check plate stack first (most robust).  Require the plate size to
            # match the main genotype count (probe_size or full num_genotype) so
            # that subset plates — e.g. the binding-observer's genotype plate,
            # which covers only calibration genotypes — are not mistaken for the
            # main genotype plate.
            num_genotype = self.model.data.num_genotype
            in_plate = False
            for frame in site.get("cond_indep_stack", []):
                if ("genotype" in frame.name.lower()
                        and frame.size in (probe_size, num_genotype)):
                    dim_map[name] = frame.dim
                    in_plate = True
                    break

            if in_plate:
                continue

            # Skip sites that belong to a non-genotype plate at the genotype
            # dim — they are indexed by a different axis (e.g. condition_pre)
            # and must NOT be treated as genotype-indexed.
            in_other_plate_at_geno_dim = any(
                frame.dim == genotype_dim
                for frame in site.get("cond_indep_stack", [])
            )
            if in_other_plate_at_geno_dim:
                continue

            # Fallback for deterministics computed outside the plate
            # but matching the genotype size at the expected dimension.
            val = site["value"]
            if hasattr(val, "shape"):
                # Handle negative indexing for the dimension check
                actual_dim = genotype_dim if genotype_dim >= 0 else len(val.shape) + genotype_dim
                if actual_dim >= 0 and actual_dim < len(val.shape):
                    if val.shape[actual_dim] == probe_size:
                        dim_map[name] = genotype_dim

        return dim_map

    @staticmethod
    def _genotype_chunk_indices(total_num_genotypes, forward_batch_size):
        """
        Build a (num_chunks, forward_batch_size) array of genotype indices
        covering ``total_num_genotypes``, padding the final chunk (if
        ``total_num_genotypes`` is not evenly divisible) by repeating the
        last valid genotype index. Padding is trimmed back off after the
        forward pass by :meth:`_concat_genotype_chunks`, so the duplicated
        rows never reach the caller.

        Returns
        -------
        jnp.ndarray, shape (num_chunks, forward_batch_size)
        """
        num_chunks = -(-total_num_genotypes // forward_batch_size)
        padded_total = num_chunks * forward_batch_size
        idx = jnp.arange(total_num_genotypes)
        pad = padded_total - total_num_genotypes
        if pad > 0:
            pad_idx = jnp.full((pad,), total_num_genotypes - 1, dtype=idx.dtype)
            idx = jnp.concatenate([idx, pad_idx])
        return idx.reshape(num_chunks, forward_batch_size)

    @staticmethod
    def _concat_genotype_chunks(chunk_list, axis, total_size):
        """
        Concatenate per-chunk numpy arrays along ``axis`` and trim padding.

        Parameters
        ----------
        chunk_list : list of np.ndarray
            Per-chunk arrays in index order (each has ``forward_batch_size``
            elements along ``axis``; the last chunk may be padded).
        axis : int
            Genotype axis (from ``dim_map``; may be negative).
        total_size : int
            True (unpadded) genotype count to trim to.

        Returns
        -------
        np.ndarray
        """
        merged = np.concatenate(chunk_list, axis=axis)
        pos = axis if axis >= 0 else merged.ndim + axis
        slices = [slice(None)] * merged.ndim
        slices[pos] = slice(0, total_size)
        return merged[tuple(slices)]

    def _batch_positional_latents(self):
        """
        Latent sites whose value has one entry per batch *position* rather
        than per genotype (e.g. ``noise/beta``'s ``{name}_dist``).

        Every other genotype-indexed latent is sampled at library size and
        sliced to the batch inside its component with ``batch_idx``, so a
        forward pass over a genotype chunk must hand it the full,
        library-ordered value -- exactly as training does. Slicing it to the
        chunk beforehand breaks components that index their parameters by
        library position (``hill_geno``, ``thermo.*``, ``categorical_geno``):
        every chunk after the first reads past the sliced array, and JAX
        clamps the index instead of raising. Only the batch-positional
        latents found here are sliced to the chunk.
        """
        if not hasattr(self, "_batch_positional_cache"):
            self._batch_positional_cache = frozenset(
                find_orchestrator_batch_dependent_latents(self.model))
        return self._batch_positional_cache

    def _build_genotype_chunk_scanner(self, dim_map, sites_to_save):
        """
        Build a JIT-compiled forward-pass function for a single genotype chunk.

        Returns a function ``chunk_fn(data, latents, key, batch_indices)``
        that runs the model forward pass for the genotype batch given by
        ``batch_indices`` and returns ``(new_key, result_dict)``.

        The function is compiled once by ``jax.jit`` on first call and
        reused across all chunks and all outer sampling-batch iterations,
        eliminating per-chunk re-tracing overhead while keeping GPU memory
        usage to a single chunk at a time.  (A ``lax.scan`` over all chunks
        would pre-allocate all chunk outputs simultaneously on device,
        causing OOM on large datasets.)

        Parameters
        ----------
        dim_map : dict
            Site name -> genotype-axis index, as returned by
            `_get_genotype_dim_map`.
        sites_to_save : list of str or None
            If given, restricts the per-chunk output to these sites.

        Returns
        -------
        callable
            ``chunk_fn(data, latents, key, batch_indices)
            -> (new_key, chunk_dict)``
        """
        model_fn = self.model.jax_model
        get_batch = self.model.get_batch
        priors = self.model.priors
        batch_positional = self._batch_positional_latents()

        @jax.jit
        def chunk_fn(data, latents, key, batch_indices):
            # Output: every genotype-indexed site, sliced to this chunk.
            batch_latents = {
                k: jnp.take(v, batch_indices, axis=dim_map[k]) if k in dim_map else v
                for k, v in latents.items()
            }
            # Model input: library-sized latents stay whole (components slice
            # them with batch_idx); only batch-positional ones are sliced.
            model_latents = {
                k: batch_latents[k] if k in batch_positional else v
                for k, v in latents.items()
            }
            batch_data = get_batch(data, batch_indices)

            key, subkey = jax.random.split(key)
            forward_sampler = Predictive(model_fn, posterior_samples=model_latents)
            batch_pred = forward_sampler(subkey, priors=priors, data=batch_data)

            # Predictions take precedence over latents of the same name.
            merged = dict(batch_latents)
            merged.update(batch_pred)
            if sites_to_save is not None:
                merged = {k: v for k, v in merged.items() if k in sites_to_save}

            return key, merged

        return chunk_fn

    def get_posteriors(self,
                       svi,
                       svi_state,
                       out_prefix,
                       num_posterior_samples=10000,
                       sampling_batch_size=100,
                       forward_batch_size=512,
                       sites_to_save=None):
        """
        Generate and save posterior samples using the trained guide.

        Uses `numpyro.infer.Predictive` to sample from the posterior
        distribution defined by the guide and parameters. Handles large
        datasets by batching predictions and writing to disk (HDF5). The
        forward pass over genotype batches uses a JIT-compiled per-chunk
        function (see `_build_genotype_chunk_scanner`) so that `Predictive`
        is traced only once regardless of the number of chunks, and GPU
        memory holds at most one chunk at a time.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            The SVI object being used for inference.
        svi_state : Any
            The current state of the SVI object (optimizer state).
        out_prefix : str
            Root name for the output file.
        num_posterior_samples : int, optional
            Number of posterior samples to draw (default 10000).
        sampling_batch_size : int, optional
            Batch size for generating posterior samples of latent parameters
            (default 100).
        forward_batch_size : int, optional
            Batch size for calculating forward predictions (default 512).
        sites_to_save : list of str or None, optional
            If given, only these site names are written to the HDF5 file.
            If None (default), all sites are saved. Use this to reduce output
            file size when only a subset of parameters is needed.
        """

        guide = svi.guide
        params = jax.device_put(svi.get_params(svi_state))
        data_on_gpu = jax.device_put(self.model.data)

        # Get the mapping of site names to genotype dimension
        dim_map = self._get_genotype_dim_map()

        total_num_genotypes = self.model.data.num_genotype

        # Adjust sampling_batch_size if smaller than num_posterior_samples
        sampling_batch_size = min(sampling_batch_size, num_posterior_samples)
        num_latent_batches = -(-num_posterior_samples // sampling_batch_size)

        # Create a full-batch data object for sampling latents
        all_indices = jnp.arange(total_num_genotypes)
        full_data = self.model.get_batch(data_on_gpu, all_indices)

        latent_sampler = Predictive(guide,
                                    params=params,
                                    num_samples=sampling_batch_size)

        # Build the genotype-chunk index blocks and the compiled forward
        # scanner once; both are reused, unchanged, across every posterior
        # sampling batch below.
        indices_2d = self._genotype_chunk_indices(total_num_genotypes, forward_batch_size)
        chunk_fn = self._build_genotype_chunk_scanner(dim_map, sites_to_save)

        # Prepare HDF5 file
        h5_file = f"{out_prefix}_posterior.h5"

        samples_written = 0
        with h5py.File(h5_file, 'w') as hf:

            for batch_i in tqdm(range(num_latent_batches), desc="sampling posterior"):

                # Sample the guide posterior
                post_key = self.get_key()
                latent_samples = latent_sampler(post_key,
                                                priors=self.model.priors,
                                                data=full_data)

                # Drop autoguide auxiliary sites (e.g. AutoContinuous's
                # flattened "_auto_latent", shape (samples, D)); they are not
                # model sites.
                latent_samples = {k: v for k, v in latent_samples.items()
                                  if not k.startswith("_")}

                # Forward pass: iterate over genotype chunks with the JIT-compiled
                # chunk_fn.  Each chunk is computed, transferred to CPU, and
                # discarded from GPU before the next chunk runs, keeping GPU
                # memory usage to a single chunk at a time.
                forward_key = self.get_key()
                chunk_outputs = {}
                for chunk_indices in indices_2d:
                    forward_key, chunk_result = chunk_fn(
                        data_on_gpu, latent_samples, forward_key, chunk_indices
                    )
                    for k, v in chunk_result.items():
                        chunk_outputs.setdefault(k, []).append(np.asarray(v))

                # Concatenate chunks along the genotype axis; global (non-dim_map)
                # sites are identical across chunks so take the first chunk only.
                this_batch_results = {}
                for k, chunks in chunk_outputs.items():
                    if k in dim_map:
                        this_batch_results[k] = self._concat_genotype_chunks(
                            chunks, dim_map[k], total_num_genotypes
                        )
                    else:
                        this_batch_results[k] = chunks[0]

                # latent_sampler always draws a full sampling_batch_size of
                # samples (fixed at construction); trim the final batch down
                # to the number of samples actually still needed when
                # num_posterior_samples isn't evenly divisible by
                # sampling_batch_size.
                this_batch_size = min(sampling_batch_size,
                                      num_posterior_samples - samples_written)
                this_batch_results = {
                    k: v[:this_batch_size] for k, v in this_batch_results.items()
                }

                # Write to file
                batch_size_actual = next(iter(this_batch_results.values())).shape[0]
                for k, v in this_batch_results.items():
                    if k not in hf:
                        # Create dataset on first write
                        maxshape = (num_posterior_samples,) + v.shape[1:]
                        chunks = _safe_chunks(min(sampling_batch_size, 100), v.shape[1:], v.dtype)
                        hf.create_dataset(k, shape=maxshape, dtype=v.dtype,
                                          chunks=chunks,
                                          compression="gzip", compression_opts=4)

                    hf[k][samples_written:samples_written + batch_size_actual] = v

                samples_written += batch_size_actual

            # Add metadata to HDF5 file
            hf.attrs["num_samples"] = samples_written

            # Force flush to disk to avoid read issues on cluster file systems
            hf.flush()

    def get_key(self):
        """
        Get a new JAX PRNG key, splitting the main key.

        Returns
        -------
        jax.random.PRNGKey
            A new, unique PRNG key.
        """

        new_key, self._main_key = jax.random.split(self._main_key)
        return new_key
    

    def _jitter_init_parameters(self,init_params,init_param_jitter):
        """
        Apply multiplicative log-normal jitter to initial parameter values.

        This method perturbs each parameter in the `init_params` dictionary by multiplying
        it by a log-normal random variable with standard deviation `init_param_jitter`.
        This is useful for breaking symmetry and improving optimization robustness.

        Parameters
        ----------
        init_params : dict
            Dictionary of initial parameter values, where each value is a scalar or
            NumPy/JAX array.
        init_param_jitter : float
            Standard deviation of the log-normal noise to apply. If set to 0, no jitter
            is applied and the original parameters are returned.

        Returns
        -------
        dict
            Dictionary of jittered initial parameter values, with the same keys as
            `init_params`.

        Notes
        -----
        The jitter is applied independently to each parameter (and each element if
        the parameter is an array), using a normal random variable in the exponent:
        `param = param * exp(noise * init_param_jitter)`, where `noise` is drawn
        from a standard normal distribution.
        """

        # No jitter requested
        if init_param_jitter == 0:
            return init_params 

        # Go through each parameter
        for p in init_params:

            # Key for randomness
            jitter_key = self.get_key()

            # Get noise of the right dimensionality
            if jnp.isscalar(init_params[p]):
                noise = random.normal(jitter_key)
            else:
                noise = random.normal(jitter_key,shape=init_params[p].shape)
            
            # add noise to init_params[p]
            init_params[p] = init_params[p]*jnp.exp(noise*init_param_jitter)

        return init_params


    def _write_checkpoint(self,svi_state,out_prefix):
        """
        Atomically save the SVI state and PRNG key to a dill pickle file.

        Parameters
        ----------
        svi_state : Any
            The SVI state (e.g., from `svi.update`).
        out_prefix : str
            Root name for the output checkpoint file.
        """

        host_svi_state = jax.device_get(svi_state)

        out_dict = {"main_key":self._main_key,
                    "svi_state":host_svi_state,
                    "current_step":self._current_step,
                    "step_size": self._checkpoint_step_size(),
                    "convergence": self._convergence_state(),
                    "guide_type":self._guide_type,
                    "guide_kwargs":self._guide_kwargs}

        tmp_checkpoint_file = f"{out_prefix}_checkpoint.tmp.pkl"

        checkpoint_file = f"{out_prefix}_checkpoint.pkl"

        # Atomic save
        with open(tmp_checkpoint_file,'wb') as f:
            dill.dump(out_dict,f)
        os.replace(tmp_checkpoint_file,
                   checkpoint_file)

    def _write_epoch_checkpoint(self, svi_state, epoch):
        """
        Save a numbered epoch checkpoint to the ``checkpoints/`` directory.

        Parameters
        ----------
        svi_state : Any
            The SVI state to save.
        epoch : int
            Current epoch number, used to name the file.

        Raises
        ------
        FileExistsError
            If the target checkpoint file already exists.
        """

        epoch_file = os.path.join(self._epoch_checkpoints_dir,
                                  f"{epoch:07d}_checkpoint.pkl")

        if os.path.exists(epoch_file):
            raise FileExistsError(
                f"Epoch checkpoint '{epoch_file}' already exists. Delete the "
                "file or change out_prefix to avoid overwriting a previous run."
            )

        host_svi_state = jax.device_get(svi_state)
        out_dict = {"main_key": self._main_key,
                    "svi_state": host_svi_state,
                    "current_step": self._current_step,
                    "step_size": self._checkpoint_step_size(),
                    "convergence": self._convergence_state(),
                    "guide_type": self._guide_type,
                    "guide_kwargs": self._guide_kwargs}

        tmp_file = f"{epoch_file}.tmp"
        with open(tmp_file, "wb") as f:
            dill.dump(out_dict, f)
        os.replace(tmp_file, epoch_file)

    def restore_svi_from_checkpoint(self, checkpoint_file, init_params=None):
        """
        Rebuild the SVI object recorded in a checkpoint and restore its state,
        without running any optimization steps or convergence checks.

        The guide is rebuilt from the checkpoint's ``guide_type`` and
        ``guide_kwargs``.  Checkpoints written before those were recorded are
        treated as component-guide checkpoints (the only SVI guide then).

        ``svi.init()`` must be called at least once to wire up ``constrain_fn``
        on the SVI object before ``svi.get_params()`` can be used.  The state
        produced by ``init()`` is immediately discarded and replaced with the
        one loaded from the checkpoint.

        Parameters
        ----------
        checkpoint_file : str
            Path to the checkpoint .pkl file produced by tfs-fit-model.
        init_params : dict or None, optional
            Initial parameter values forwarded to ``svi.init()`` for a
            component guide.  The values are never used in inference (the
            checkpoint overwrites them), but they must be structurally valid
            for the model.  Ignored for autoguides, whose parameter names
            differ from the component guide's.

        Returns
        -------
        svi : numpyro.infer.SVI
            Initialized SVI object with ``constrain_fn`` populated.
        svi_state : numpyro.infer.svi.SVIState
            Optimizer state restored from the checkpoint.
        """
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = dill.load(f)
        guide_type = checkpoint_data.get("guide_type") or "component"
        guide_kwargs = checkpoint_data.get("guide_kwargs") or None

        svi = self.setup_svi(guide_type=guide_type, guide_kwargs=guide_kwargs)
        if self._guide_type != "component":
            init_params = None

        # init() is required to populate svi.constrain_fn; the resulting state
        # is thrown away — the checkpoint state is used instead.
        data_on_gpu = jax.device_put(self.model.data)
        batch_idx = jax.device_put(self.model.get_random_idx())
        batch_data = self.model.get_batch(data_on_gpu, batch_idx)
        init_key = self.get_key()
        svi.init(init_key, init_params=init_params,
                 priors=self.model.priors, data=batch_data)

        svi_state = self._restore_checkpoint(checkpoint_file)
        return svi, svi_state

    def _restore_checkpoint(self,checkpoint_file):
        """
        Load an SVI state and PRNG key from a checkpoint file.

        Parameters
        ----------
        checkpoint_file : str
            Path to the checkpoint .pkl file.

        Returns
        -------
        Any
            The restored SVI state.
        """
    
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = dill.load(f)
    
        svi_state = checkpoint_data['svi_state'] 
        self._main_key = checkpoint_data['main_key']
        if 'current_step' in checkpoint_data:
            self._current_step = checkpoint_data['current_step']
        # Convergence stage (step size, plateau count); absent from
        # checkpoints written before step-size cuts, which resume at the
        # configured step size.
        self._monitor_state = checkpoint_data.get('convergence')

        if not isinstance(svi_state,numpyro.infer.svi.SVIState):
            raise ValueError(
                f"checkpoint_file {checkpoint_file} does not appear to have a saved svi_state"
            )

        return svi_state

    def _write_losses(self,losses,out_prefix):
        """
        Write losses to binary (.bin) and text (.txt) files. 

        Parameters
        ----------
        losses : list
            A list of loss values from the recent optimization interval.
        out_prefix : str
            Root name for the output files.
        """

        # Name of output file
        losses_file = f"{out_prefix}_losses.bin"
        readable_losses_file = f"{out_prefix}_losses.txt"

        # At the start of a run, remove stale files and write the text header.
        if self._current_step == 0:

            if os.path.exists(losses_file):
                os.remove(losses_file)

            if os.path.exists(readable_losses_file):
                os.remove(readable_losses_file)

            with open(readable_losses_file, "w") as f:
                f.write("epoch,loss,step,step_size\n")
                f.flush()
                os.fsync(f.fileno())

        # No losses to write this iteration, continue
        if len(losses) == 0:
            return

        epoch = self._current_step // self._iterations_per_epoch

        # Open in append binary mode
        with open(losses_file, "ab") as f:
            np.array(losses).tofile(f)
            f.flush()
            os.fsync(f.fileno())

        # Write a human-readable losses file: epoch, median loss of the block,
        # step, step size.
        step_size = self._checkpoint_step_size()
        with open(readable_losses_file, "a") as f:
            f.write(f"{epoch},{np.median(losses)},{self._current_step},"
                    f"{step_size}\n")
            f.flush()
            os.fsync(f.fileno())

    _CONVERGENCE_COLUMNS = ("step", "epoch", "step_size", "loss", "loss_drop",
                            "loss_drop_se", "loss_t", "loss_improving",
                            "worst_param", "worst_param_excess",
                            "worst_param_drift", "params_moving", "plateau",
                            "plateau_count", "decision")

    def _write_convergence(self, record, out_prefix):
        """
        Append one convergence-window record to ``{out_prefix}_convergence.csv``.

        With ``record=None`` (start of a run) the file is created with its
        header -- replacing any old file -- on a fresh run, and left alone on
        a resume.
        """

        path = f"{out_prefix}_convergence.csv"
        if record is None:
            if self._current_step == 0 or not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(",".join(self._CONVERGENCE_COLUMNS) + "\n")
            return

        row = dict(record)
        row["epoch"] = row["step"] // self._iterations_per_epoch
        with open(path, "a") as f:
            f.write(",".join(str(row.get(c, "")) for c in
                             self._CONVERGENCE_COLUMNS) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _checkpoint_step_size(self):
        """Current constant step size, or None for a schedule/unknown."""
        if self._monitor is not None and np.isfinite(self._monitor.step_size):
            return self._monitor.step_size
        if self._step_size is not None and not callable(self._step_size):
            return float(self._step_size)
        return None

    def _convergence_state(self):
        """Monitor state for checkpoints (None before any optimization)."""
        if self._monitor is None:
            return self._monitor_state
        return self._monitor.state_dict()




    def write_params(self,params,out_prefix):
        """
        Write parameters to an .npz file.
        
        Parameters
        ----------
        params : dict
            dictionary of parameters
        out_prefix : str
            string to append to front of output file
        """

        tmp_out_file = f"{out_prefix}_params.tmp.npz"
        out_file = f"{out_prefix}_params.npz"

        np.savez_compressed(tmp_out_file,**params)

        os.replace(tmp_out_file,out_file)        

    def _map_params_to_constrained(self, map_params):
        """
        Convert MAP guide parameters to constrained natural-parameter space.

        Strips the ``_auto_loc`` suffix added by AutoDelta, then applies the
        per-site bijection (e.g. softplus for positive-constrained sites) to
        produce a dict of constrained values, one entry per latent site.

        Parameters
        ----------
        map_params : dict
            Parameter dict from an AutoDelta guide state, keys follow the
            ``{site}_auto_loc`` convention, values are in unconstrained space.

        Returns
        -------
        dict
            Constrained parameters keyed by site name (no ``_auto_loc`` suffix).
            Each value has shape ``(*site_shape,)`` — a single-sample point.
        """
        from numpyro.distributions.transforms import biject_to

        data_on_gpu = jax.device_put(self.model.data)
        total_num_genotypes = self.model.data.num_genotype
        all_indices = jnp.arange(total_num_genotypes)
        full_data = self.model.get_batch(data_on_gpu, all_indices)
        model_kwargs = {"priors": self.model.priors, "data": full_data}

        unconstrained = {
            k[: -len("_auto_loc")]: jnp.array(v)
            for k, v in map_params.items()
            if k.endswith("_auto_loc")
        }

        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        model_trace = traced_model.get_trace(**model_kwargs)

        site_transforms = {
            name: biject_to(site["fn"].support)
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }

        return {
            k: site_transforms[k](v) if k in site_transforms else v
            for k, v in unconstrained.items()
        }

    @staticmethod
    def _check_library_sized_latents(latents, dim_map, total_num_genotypes):
        """
        Raise if any genotype-indexed MAP latent is not library-sized.

        A batch-sized latent (a MAP checkpoint written before per-genotype
        latents were sampled at library size, or a component that cannot be
        -- see inference/batch_safety.py) holds values fit per mini-batch
        position, aliased across genotypes.  Refuse it rather than reuse one
        genotype's value for another.  ``dim_map`` axes are negative, so
        ``latents`` may or may not carry a leading sample axis.
        """
        for k, v in latents.items():
            if k in dim_map and v.shape[dim_map[k]] != total_num_genotypes:
                raise ValueError(
                    f"MAP latent '{k}' has {v.shape[dim_map[k]]} entries along "
                    f"the genotype axis but the library has "
                    f"{total_num_genotypes} genotypes. Its values were fit per "
                    f"mini-batch position, not per genotype, so they cannot "
                    f"be mapped back onto genotypes. Refit the model."
                )

    def get_map_posteriors(self,
                           map_params,
                           out_prefix,
                           forward_batch_size=512,
                           sites_to_save=None):
        """
        Generate and save a single-sample posterior at the MAP point.

        Converts the MAP guide parameters to constrained space and runs one
        forward pass of the generative model, writing a 1-sample HDF5 file in
        the same format produced by :meth:`get_posteriors`,
        :meth:`get_laplace_posteriors`, and :meth:`get_nuts_posteriors`.

        This is useful for checking whether the MAP solution is consistent
        with the observed data without introducing Hessian-based uncertainty.
        Unlike the Laplace approximation, no sampling or Hessian computation
        is required: the output contains exactly the predictions at the MAP
        point.

        Parameters
        ----------
        map_params : dict
            Parameter dict from a MAP (AutoDelta) optimizer state, as
            returned by ``svi.get_params(svi_state)``.  Keys follow the
            ``{site}_auto_loc`` convention; values are in unconstrained space.
        out_prefix : str
            Root name for the output file (written as
            ``{out_prefix}_posterior.h5``).
        forward_batch_size : int, optional
            Number of genotypes to process per forward-model batch
            (default 512).
        sites_to_save : list of str or None, optional
            If given, only these site names are written to the HDF5 file.
            If None (default), all sites are saved.
        """
        data_on_gpu = jax.device_put(self.model.data)
        total_num_genotypes = self.model.data.num_genotype
        dim_map = self._get_genotype_dim_map()

        constrained = self._map_params_to_constrained(map_params)
        # Add a leading sample dimension: (*shape) → (1, *shape)
        latent_samples = {k: jnp.expand_dims(v, 0) for k, v in constrained.items()}

        self._check_library_sized_latents(latent_samples, dim_map,
                                          total_num_genotypes)

        h5_file = f"{out_prefix}_posterior.h5"

        batch_collector = {}
        for start_idx in range(0, total_num_genotypes, forward_batch_size):

            end_idx = min(start_idx + forward_batch_size, total_num_genotypes)
            batch_indices = jnp.arange(start_idx, end_idx)

            batch_latents = {
                k: jnp.take(v, batch_indices, axis=dim_map[k])
                if k in dim_map else v
                for k, v in latent_samples.items()
            }
            # Library-sized latents stay whole for the model (see
            # _batch_positional_latents); batch_latents is only the output.
            batch_positional = self._batch_positional_latents()
            model_latents = {
                k: batch_latents[k] if k in batch_positional else v
                for k, v in latent_samples.items()
            }

            batch_data = self.model.get_batch(data_on_gpu, batch_indices)
            forward_sampler = Predictive(self.model.jax_model,
                                         posterior_samples=model_latents)
            pred_key = self.get_key()
            batch_pred = forward_sampler(pred_key,
                                         priors=self.model.priors,
                                         data=batch_data)

            for k, v in batch_pred.items():
                if sites_to_save is not None and k not in sites_to_save:
                    continue
                batch_collector.setdefault(k, []).append(jax.device_get(v))

            for k, v in batch_latents.items():
                if k in batch_pred:
                    continue
                if sites_to_save is not None and k not in sites_to_save:
                    continue
                if k not in dim_map:
                    if start_idx == 0:
                        batch_collector.setdefault(k, []).append(jax.device_get(v))
                else:
                    batch_collector.setdefault(k, []).append(jax.device_get(v))

        results = {}
        for k, v_list in batch_collector.items():
            if k in dim_map:
                results[k] = np.concatenate(v_list, axis=dim_map[k])
            else:
                results[k] = v_list[0]

        with h5py.File(h5_file, "w") as hf:
            for k, v in results.items():
                chunks = _safe_chunks(min(1, v.shape[0]), v.shape[1:], v.dtype)
                hf.create_dataset(k, data=v, chunks=chunks,
                                  compression="gzip", compression_opts=4)
            hf.attrs["num_samples"] = 1
            hf.flush()

    def _chunked_hessian(self, pe_fn, flat_map, chunk_size):
        """
        Compute the full D×D Hessian of ``pe_fn`` at ``flat_map`` in row-chunks.

        ``jax.hessian`` vmaps all D basis-vector JVPs simultaneously, which
        allocates D × (gradient intermediates) on the accelerator at once —
        several GB for models with O(10³) parameters.  This helper limits peak
        device memory to ``chunk_size × (gradient intermediates)`` by looping
        over row-batches and immediately transferring each chunk to CPU.

        Parameters
        ----------
        pe_fn : callable
            Scalar function of a flat parameter vector.
        flat_map : jnp.ndarray, shape (D,)
            Point at which to evaluate the Hessian.
        chunk_size : int
            Number of Hessian rows to compute per device batch.  Smaller
            values use less device memory but require more iterations.
            ``chunk_size=64`` is a safe default for most GPU sizes.

        Returns
        -------
        numpy.ndarray, shape (D, D), dtype float64
            The full Hessian matrix on CPU.
        """
        D = flat_map.shape[0]
        grad_fn = jax.grad(pe_fn)
        rows = []
        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            basis_chunk = jnp.eye(D, dtype=flat_map.dtype)[start:end]
            _, H_chunk = jax.vmap(
                lambda v: jax.jvp(grad_fn, (flat_map,), (v,))
            )(basis_chunk)
            rows.append(np.array(H_chunk, dtype=np.float64))
        return np.concatenate(rows, axis=0)

    def compute_hessian_sigmas(self, map_params, hessian_chunk_size=64):
        """
        Compute per-site Hessian-based MAP values and sigmas in *constrained*
        parameter space.

        Used by the calibration pre-fit to extract uncertainty estimates on
        the linking-function hyperparameters at the MAP point, without
        needing to draw a full Laplace posterior.

        The Hessian is computed in unconstrained space (where all variables
        are real-valued).  For each sample site, the constrained MAP value
        and an elementwise constrained-space sigma are returned.  For
        constrained supports (e.g. ``HalfNormal`` → ``positive`` →
        ``ExpTransform``) the unconstrained sigma is propagated through the
        bijection's elementwise Jacobian via the delta method:

            sigma_constrained = |dT/dx| * sigma_unconstrained

        For unconstrained supports (``Normal`` → identity) this reduces to
        ``sigma_constrained = sigma_unconstrained``.

        Negative Hessian eigenvalues (indicating saddle points) are clamped
        to ``1e-3`` before inversion, matching the behaviour of
        :meth:`get_laplace_posteriors`.

        Parameters
        ----------
        map_params : dict
            Parameter dict from a MAP (AutoDelta) optimizer state, as
            returned by ``svi.get_params(svi_state)``.  Keys follow the
            ``{site}_auto_loc`` convention; values are in unconstrained
            space.
        hessian_chunk_size : int, optional
            Number of Hessian rows to compute per device batch (default 64).
            Reduce if you hit device OOM during the Hessian computation.

        Returns
        -------
        dict[str, dict]
            Maps each sample-site name to a dict with two arrays:

            * ``"map"``    : constrained MAP value (same shape as the site)
            * ``"sigma"``  : elementwise constrained-space 1-sigma uncertainty

            Both are returned as plain ``numpy`` arrays.
        """
        from numpyro.infer.util import potential_energy
        from numpyro.distributions.transforms import biject_to
        import jax.flatten_util

        data_on_gpu = jax.device_put(self.model.data)
        total_num_genotypes = self.model.data.num_genotype
        all_indices = jnp.arange(total_num_genotypes)
        full_data = self.model.get_batch(data_on_gpu, all_indices)
        model_kwargs = {"priors": self.model.priors, "data": full_data}

        # Strip _auto_loc suffix → unconstrained site-level param dict
        unconstrained = {
            k[: -len("_auto_loc")]: jnp.array(v)
            for k, v in map_params.items()
            if k.endswith("_auto_loc")
        }

        if len(unconstrained) == 0:
            return {}

        # Flatten to a single vector for Hessian computation.  We need to
        # remember the per-site shapes / offsets so we can pull the
        # diagonal back into per-site sigma arrays.
        flat_map, unravel = jax.flatten_util.ravel_pytree(unconstrained)
        D = flat_map.shape[0]

        def pe_fn(flat_p):
            return potential_energy(
                self.model.jax_model, [], model_kwargs, unravel(flat_p)
            )

        print(f"Computing Hessian for {D} parameters "
              f"(chunk_size={hessian_chunk_size}) ...", flush=True)
        H_np = self._chunked_hessian(pe_fn, flat_map, hessian_chunk_size)

        # Project Hessian to the PD cone in float64; we only need the diagonal
        # of the inverse, which is sigma^2 per element.
        eigenvalues_np, eigenvectors_np = np.linalg.eigh(H_np)
        eigenvalues_pd = np.maximum(eigenvalues_np, 1e-3)
        cov_diag_np = np.einsum(
            "ij,j,ij->i",
            eigenvectors_np,
            1.0 / eigenvalues_pd,
            eigenvectors_np,
        )
        sigma_unconstrained_flat = np.sqrt(np.maximum(cov_diag_np, 0.0))

        # Unravel the unconstrained sigmas back to per-site shape.
        sigma_unconstrained = unravel(jnp.array(sigma_unconstrained_flat,
                                                 dtype=flat_map.dtype))

        # Get the per-site bijection from a model trace.  For unconstrained
        # supports (Normal) this is the identity transform.
        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        model_trace = traced_model.get_trace(**model_kwargs)
        site_transforms = {
            name: biject_to(site["fn"].support)
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }

        out = {}
        for name, x_unc in unconstrained.items():
            transform = site_transforms.get(name)
            sigma_unc = sigma_unconstrained[name]
            if transform is None:
                # No bijection available — assume identity.
                map_constrained = x_unc
                sigma_constrained = sigma_unc
            else:
                map_constrained = transform(x_unc)
                # Elementwise Jacobian magnitude.  log_abs_det_jacobian
                # gives sum across event dims; we want per-element |dT/dx|
                # which (for diagonal transforms) = exp(log|dy/dx|).  Use
                # jax.grad of a scalar projection so we don't depend on the
                # transform's internal API.
                sigma_constrained = jnp.abs(
                    jax.vmap(lambda v: jax.grad(transform)(v))(
                        x_unc.reshape(-1)
                    )
                ).reshape(x_unc.shape) * sigma_unc

            out[name] = {
                "map": np.asarray(map_constrained),
                "sigma": np.asarray(sigma_constrained),
            }

        return out

    def _get_site_names(self,target_sites="deterministic"):
        """
        Dry-runs a NumPyro model to extract site names programmatically.
        
        Parameters
        ----------
        target_sites : str or list of str
            The type of sites to extract (e.g., 'deterministic', 'sample').
            
        Returns
        -------
        list
            List of site names found in the model trace.
        """

        # Make sure target_types is a list
        if isinstance(target_sites,str):
            target_sites = [target_sites]

        # Seed the model so it runs deterministically and trace it to capture
        # the execution flow. 
        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        
        # Run the traced model
        model_trace = traced_model.get_trace(data=self.model.data,
                                             priors=self.model.priors)
        
        # Get all sites matching target_sites
        site_names = [
            name for name, site_info in model_trace.items()
            if site_info["type"] in target_sites
        ]
        
        return site_names

    def get_laplace_posteriors(self,
                               map_params,
                               out_prefix,
                               num_posterior_samples=10000,
                               sampling_batch_size=100,
                               forward_batch_size=512,
                               hessian_chunk_size=64,
                               sites_to_save=None):
        """
        Generate posterior samples from a MAP solution using the Laplace approximation.

        Computes the Hessian of the negative log-joint at the MAP point, inverts
        it to obtain a covariance matrix, and draws samples from the resulting
        multivariate Gaussian. Samples are pushed through the generative model
        and written to an HDF5 file in the same format produced by
        ``get_posteriors``.

        Parameters
        ----------
        map_params : dict
            Parameter dict from a MAP (AutoDelta) optimizer state, as returned
            by ``svi.get_params(svi_state)``. Keys follow the
            ``{site}_auto_loc`` convention used by AutoDelta; values are in
            the unconstrained parameter space.
        out_prefix : str
            Root name for the output file (written as
            ``{out_prefix}_posterior.h5``).
        num_posterior_samples : int, optional
            Number of posterior samples to draw (default 10000).
        sampling_batch_size : int, optional
            Number of latent samples to draw per batch (default 100).
        forward_batch_size : int, optional
            Number of genotypes to process per forward-model batch (default 512).
        hessian_chunk_size : int, optional
            Number of Hessian rows to compute per device batch (default 64).
            Reduce if you hit device OOM during the Hessian computation.
        sites_to_save : list of str or None, optional
            If given, only these site names are written to the HDF5 file.
            If None (default), all sites are saved.

        Notes
        -----
        Hessian inversion is O(D²) in memory and O(D³) in compute, where D is
        the total number of unconstrained parameters.  The Hessian is computed
        in row-chunks (see ``hessian_chunk_size``) to bound peak device memory
        to ``chunk_size × (gradient intermediates)`` rather than
        ``D × (gradient intermediates)``.
        """
        from numpyro.infer.util import potential_energy
        from numpyro.distributions.transforms import biject_to
        import jax.flatten_util

        data_on_gpu = jax.device_put(self.model.data)
        total_num_genotypes = self.model.data.num_genotype
        dim_map = self._get_genotype_dim_map()

        all_indices = jnp.arange(total_num_genotypes)
        full_data = self.model.get_batch(data_on_gpu, all_indices)
        model_kwargs = {"priors": self.model.priors, "data": full_data}

        # Strip _auto_loc suffix → unconstrained site-level param dict
        unconstrained = {
            k[: -len("_auto_loc")]: jnp.array(v)
            for k, v in map_params.items()
            if k.endswith("_auto_loc")
        }

        # Refuse aliased (batch-sized) latents before the costly Hessian.
        self._check_library_sized_latents(unconstrained, dim_map,
                                          total_num_genotypes)

        # Flatten to a single vector for Hessian computation
        flat_map, unravel = jax.flatten_util.ravel_pytree(unconstrained)
        D = flat_map.shape[0]
        print(f"Computing Hessian for {D} parameters "
              f"(chunk_size={hessian_chunk_size}) ...", flush=True)

        def pe_fn(flat_p):
            return potential_energy(
                self.model.jax_model, [], model_kwargs, unravel(flat_p)
            )

        H_np = self._chunked_hessian(pe_fn, flat_map, hessian_chunk_size)

        # Project Hessian to the PD cone and compute the Cholesky factor of the
        # covariance in float64 (via numpy) so that sampling is numerically
        # stable even when JAX_ENABLE_X64 is not set.
        #
        # Three sources of instability in the naive float32 approach:
        #  1. Saddle-point MAP → negative Hessian eigenvalues → non-PD covariance
        #  2. Very large eigenvalues (e.g. 2.6e8) → covariance condition number
        #     ~1e11, which exceeds float32 precision (~1e7) and reintroduces
        #     negative eigenvalues after the V @ diag(1/λ) @ V^T reconstruction.
        #  3. jax.random.multivariate_normal uses Cholesky internally; if the
        #     covariance is not PD the Cholesky fails → NaN samples.
        #
        # Fix: do the eigendecomposition and Cholesky in numpy float64, clamp
        # negative eigenvalues to 1e-3 (caps max variance per direction at 1000),
        # then sample as mean + L @ z where z ~ N(0,I) in float32.
        print("Projecting Hessian to PD cone and computing Cholesky ...", flush=True)
        eigenvalues_np, eigenvectors_np = np.linalg.eigh(H_np)
        n_negative = int(np.sum(eigenvalues_np < 0))
        if n_negative > 0:
            print(f"  Warning: {n_negative} negative Hessian eigenvalues "
                  f"(min={eigenvalues_np.min():.3e}); clamping to 1e-3.",
                  flush=True)
        eigenvalues_pd = np.maximum(eigenvalues_np, 1e-3)
        cov_np = eigenvectors_np @ np.diag(1.0 / eigenvalues_pd) @ eigenvectors_np.T
        # Cholesky in float64; cast factor to float32 for the forward pass
        L_np = np.linalg.cholesky(cov_np)
        L = jnp.array(L_np, dtype=jnp.float32)

        # Get unconstrained → constrained transform for each latent site
        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        model_trace = traced_model.get_trace(**model_kwargs)

        site_transforms = {
            name: biject_to(site["fn"].support)
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        }

        sampling_batch_size = min(sampling_batch_size, num_posterior_samples)
        num_latent_batches = -(-num_posterior_samples // sampling_batch_size)

        # Build the per-chunk function once; reused across all sampling batches.
        indices_2d = self._genotype_chunk_indices(total_num_genotypes, forward_batch_size)
        chunk_fn = self._build_genotype_chunk_scanner(dim_map, sites_to_save)

        h5_file = f"{out_prefix}_posterior.h5"
        samples_written = 0

        with h5py.File(h5_file, "w") as hf:

            for _ in tqdm(range(num_latent_batches), desc="sampling posterior"):

                # Last batch may be smaller when num_posterior_samples is not
                # evenly divisible by sampling_batch_size.
                this_batch_size = min(sampling_batch_size,
                                      num_posterior_samples - samples_written)

                # Draw flat unconstrained samples: (this_batch_size, D)
                # Use mean + z @ L^T (z ~ N(0,I)) to avoid a second Cholesky
                # inside jax.random.multivariate_normal.
                sample_key = self.get_key()
                z = jax.random.normal(sample_key, shape=(this_batch_size, D))
                flat_samples = flat_map + z @ L.T

                # Unravel each sample and transform to constrained space.
                # jax.vmap(unravel) maps (N, D) → pytree of (N, *shape) arrays.
                batch_unconstrained = jax.vmap(unravel)(flat_samples)
                latent_samples = {
                    k: jax.vmap(site_transforms[k])(v) if k in site_transforms else v
                    for k, v in batch_unconstrained.items()
                }

                # Forward pass over genotype chunks; one chunk at a time on GPU.
                forward_key = self.get_key()
                chunk_outputs = {}
                for chunk_indices in indices_2d:
                    forward_key, chunk_result = chunk_fn(
                        data_on_gpu, latent_samples, forward_key, chunk_indices
                    )
                    for k, v in chunk_result.items():
                        chunk_outputs.setdefault(k, []).append(np.asarray(v))

                this_batch = {}
                for k, chunks in chunk_outputs.items():
                    if k in dim_map:
                        this_batch[k] = self._concat_genotype_chunks(
                            chunks, dim_map[k], total_num_genotypes
                        )
                    else:
                        this_batch[k] = chunks[0]

                # Write to HDF5
                batch_size_actual = next(iter(this_batch.values())).shape[0]
                for k, v in this_batch.items():
                    if k not in hf:
                        maxshape = (num_posterior_samples,) + v.shape[1:]
                        chunks = _safe_chunks(min(sampling_batch_size, 100), v.shape[1:], v.dtype)
                        hf.create_dataset(k, shape=maxshape, dtype=v.dtype,
                                          chunks=chunks,
                                          compression="gzip", compression_opts=4)
                    hf[k][samples_written: samples_written + batch_size_actual] = v

                samples_written += batch_size_actual

            hf.attrs["num_samples"] = samples_written
            hf.flush()

    def get_nuts_posteriors(self,
                            mcmc_samples,
                            out_prefix,
                            forward_batch_size=512,
                            sites_to_save=None):
        """
        Generate and save posterior predictions from NUTS MCMC samples.

        Takes posterior samples already drawn by ``run_nuts()``, runs the
        forward model on them, and writes results to an HDF5 file in the
        same format as ``get_posteriors()``.

        Parameters
        ----------
        mcmc_samples : dict
            Posterior samples as returned by ``mcmc.get_samples()``.
            Each value has shape ``(num_samples, *site_shape)``.
        out_prefix : str
            Root name for the output file (written as
            ``{out_prefix}_posterior.h5``).
        forward_batch_size : int, optional
            Number of genotypes to process per forward-model batch
            (default 512).
        sites_to_save : list of str or None, optional
            If given, only these site names are written to the HDF5 file.
            If None (default), all sites are saved.
        """

        data_on_gpu = jax.device_put(self.model.data)
        total_num_genotypes = self.model.data.num_genotype
        dim_map = self._get_genotype_dim_map()

        first_val = next(iter(mcmc_samples.values()))
        num_samples = first_val.shape[0]

        h5_file = f"{out_prefix}_posterior.h5"

        # Forward pass over genotype chunks using the JIT-compiled per-chunk
        # function so Predictive is traced only once (not per chunk).
        indices_2d = self._genotype_chunk_indices(total_num_genotypes, forward_batch_size)
        chunk_fn = self._build_genotype_chunk_scanner(dim_map, sites_to_save)
        forward_key = self.get_key()
        chunk_outputs = {}
        for chunk_indices in indices_2d:
            forward_key, chunk_result = chunk_fn(
                data_on_gpu, mcmc_samples, forward_key, chunk_indices
            )
            for k, v in chunk_result.items():
                chunk_outputs.setdefault(k, []).append(np.asarray(v))

        results = {}
        for k, chunks in chunk_outputs.items():
            if k in dim_map:
                results[k] = self._concat_genotype_chunks(
                    chunks, dim_map[k], total_num_genotypes
                )
            else:
                results[k] = chunks[0]

        with h5py.File(h5_file, "w") as hf:
            for k, v in results.items():
                chunks = _safe_chunks(min(100, v.shape[0]), v.shape[1:], v.dtype)
                hf.create_dataset(k, data=v, chunks=chunks,
                                  compression="gzip", compression_opts=4)
            hf.attrs["num_samples"] = num_samples
            hf.flush()

    def run_nuts(self,
                 num_warmup=500,
                 num_samples=500,
                 num_chains=1,
                 target_accept_prob=0.9):
        """
        Run NUTS (No-U-Turn Sampler) MCMC on the full dataset.

        Parameters
        ----------
        num_warmup : int
            Number of warmup/adaptation steps.
        num_samples : int
            Number of posterior samples to draw.
        num_chains : int
            Number of MCMC chains.
        target_accept_prob : float
            Target acceptance probability for step-size adaptation.

        Returns
        -------
        numpyro.infer.MCMC
            The MCMC object after sampling. Call `.get_samples()` to get
            posterior samples as a dict of {site_name: jnp.array}.
        """
        from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median
        import numpyro.infer.util

        main_key = random.PRNGKey(self._seed)

        jax_model_kwargs = {
            "priors": self.model.priors,
            "data": self.model.data,
        }

        # Initialise from prior median — more robust than random for
        # hierarchical models with constrained parameters.
        init_params, _, _, _ = numpyro.infer.util.initialize_model(
            main_key,
            self.model.jax_model,
            model_args=[],
            model_kwargs=jax_model_kwargs,
            init_strategy=init_to_median,
        )
        init_strategy = init_to_value(values=init_params)

        kernel = NUTS(self.model.jax_model,
                      init_strategy=init_strategy,
                      target_accept_prob=target_accept_prob)

        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)

        run_key, _ = random.split(main_key)
        mcmc.run(run_key, **jax_model_kwargs)

        divergences = mcmc.get_extra_fields().get("diverging", None)
        if divergences is not None:
            num_div = int(jnp.sum(divergences))
            total = num_samples * num_chains
            print(f"NUTS: {num_div} divergences out of {total} samples")

        return mcmc
