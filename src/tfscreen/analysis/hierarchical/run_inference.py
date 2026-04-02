
import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoDelta
import numpy as np
import dill
from tqdm.auto import tqdm

from collections import deque
import os
import h5py
from .posteriors import load_posteriors


class RunInference:
    """
    Manages the SVI (Stochastic Variational Inference) process for a model.

    This class handles SVI setup, optimization loops, checkpointing,
    convergence checking, and posterior sample generation. It is designed
    to interface with a 'model' object that defines the Pyro/PyTorch model,
    data, and initial parameters.
    """

    def __init__(self, model, seed):
        """
        Initialize the RunInference class.

        Parameters
        ----------
        model : object
            A model object that must expose the following attributes:
            - `data`: Data object, expected to have `num_genotype`.
            - `priors`: Data object holding model priors.
            - `pyro_model` (callable): The Pyro model.
            - `pyro_model_guide` (callable): The guide for the Pyro model.
        seed : int
            Random seed for reproducibility.
        """

        required_attr = ["data",
                         "priors",
                         "pyro_model",
                         "pyro_model_guide"]
        for attr in required_attr:
            if not hasattr(model, attr):
                raise ValueError(f"`model` must have attribute {attr}")

        self.model = model
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_step = 0
        self._relative_change = np.inf

        # Calculate iterations per epoch
        num_genotypes = self.model.data.num_genotype

        # Determine batch size by dry-running get_random_idx
        init_batch_key = int(self._rng.integers(0, 2**31))
        test_idx = self.model.get_random_idx(init_batch_key, num_batches=1)
        batch_size = len(np.atleast_1d(np.asarray(test_idx)).flatten())

        self._iterations_per_epoch = int(np.ceil(num_genotypes / batch_size))

        self._patience_counter = 0

    def setup_svi(self,
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  elbo_num_particles=2,
                  guide_type="delta"):
        """
        Set up SVI.

        Parameters
        ----------
        adam_step_size : float, optional
            Learning rate for the ClippedAdam optimizer.
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.
        guide_type : str, optional
            Type of guide to use.
            - 'component': Use the guide defined in the model components.
            - 'delta': Use pyro.infer.autoguide.AutoDelta (MAP estimator).

        Returns
        -------
        pyro.infer.SVI
            An SVI object.
        """

        if guide_type == "delta":
            guide = AutoDelta(self.model.pyro_model)
        elif guide_type == "component":
            guide = self.model.pyro_model_guide
        else:
            raise ValueError(f"guide_type '{guide_type}' not recognized.")

        optimizer = ClippedAdam({"lr": adam_step_size, "clip_norm": adam_clip_norm})

        svi = SVI(self.model.pyro_model,
                  guide,
                  optimizer,
                  loss=Trace_ELBO(num_particles=elbo_num_particles))

        return svi

    def run_optimization(self,
                         svi,
                         svi_state=None,
                         init_params=None,
                         out_root="tfs",
                         convergence_tolerance=0.01,
                         convergence_window=10,
                         patience=10,
                         convergence_check_interval=2,
                         checkpoint_interval=10,
                         max_num_epochs=10000000,
                         init_param_jitter=0.1):
        """
        Run the SVI optimization loop.

        Parameters
        ----------
        svi : pyro.infer.SVI
            The SVI object from `setup_svi`.
        svi_state : dict or str, optional
            An existing param store state dict to resume from, a path to a
            checkpoint file, or None to start fresh.
        init_params : dict, optional
            Initial parameters (constrained values). Used to seed the param
            store before the first step.
        out_root : str, optional
            Root name for output files (checkpoints, losses).
        convergence_tolerance : float, optional
            Relative change threshold to declare convergence.
        convergence_window : int, optional
            Number of epochs to average over for convergence check.
        patience : int, optional
            Number of consecutive convergence checks that must meet tolerance.
        convergence_check_interval : int, optional
            Frequency (in epochs) to check for convergence.
        checkpoint_interval : int, optional
            Frequency (in epochs) to write checkpoints.
        max_num_epochs : int, optional
            Maximum number of optimization epochs to run.
        init_param_jitter : float, optional
            Amount of log-normal jitter to add to init_params (0 to disable).

        Returns
        -------
        svi_state : dict
            The final param store state.
        params : dict
            The final optimized parameters as numpy arrays.
        converged : bool
            True if the optimization converged based on the tolerance.

        Raises
        ------
        RuntimeError
            If parameters explode to NaN during optimization.
        """

        # Apply jitter to init_params
        if init_params is not None:
            init_params = self._jitter_init_parameters(init_params=init_params,
                                                       init_param_jitter=init_param_jitter)

        # Initial batch for first step
        init_batch_idx = self.model.get_random_idx()
        batch_data = self.model.get_batch(self.model.data, init_batch_idx)

        if svi_state is None:
            # Seed param store with init_params before first step
            if init_params is not None:
                for k, v in init_params.items():
                    pyro.get_param_store()[k] = torch.as_tensor(v, dtype=torch.float32)
            # First step initializes any remaining guide parameters
            svi.step(priors=self.model.priors, data=batch_data)

        elif isinstance(svi_state, str):
            if os.path.isfile(svi_state):
                ps_state = self._restore_checkpoint(svi_state)
                pyro.get_param_store().set_state(ps_state)
            else:
                raise ValueError(
                    f"svi_state '{svi_state}' is not valid"
                )
        else:
            # svi_state is a param store state dict
            pyro.get_param_store().set_state(svi_state)

        # Initialize loss file
        self._write_losses(np.array([]), out_root)

        # Loss deque for convergence checking
        deque_maxlen = 2 * convergence_window * self._iterations_per_epoch
        self._loss_deque = deque(maxlen=deque_maxlen)
        converged = False

        # Convert intervals from epochs to iterations
        check_interval_steps = convergence_check_interval * self._iterations_per_epoch
        checkpoint_interval_steps = checkpoint_interval * self._iterations_per_epoch
        total_steps = max_num_epochs * self._iterations_per_epoch

        self._next_checkpoint_step = ((self._current_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps

        # Reset patience counter
        self._patience_counter = 0

        current_optimization_step = 0
        while current_optimization_step < total_steps:

            block_size = min(check_interval_steps, total_steps - current_optimization_step)
            interval_losses = []

            for _ in range(block_size):
                batch_idx = self.model.get_random_idx()
                batch_data = self.model.get_batch(self.model.data, batch_idx)
                loss = svi.step(priors=self.model.priors, data=batch_data)
                interval_losses.append(float(loss))

            interval_losses = np.array(interval_losses)

            current_optimization_step += block_size
            self._current_step += block_size

            self._update_loss_deque(interval_losses)

            print(f"Step: {self._current_step:10d}, Loss: {interval_losses[-1]:10.5e}, Change: {self._relative_change:10.5e}, Patience: {self._patience_counter:3d}", flush=True)

            # Check for NaN explosion in parameters
            for k, v in pyro.get_param_store().items():
                v_np = np.asarray(v.detach().cpu())
                if np.any(np.isnan(v_np)):
                    raise RuntimeError(
                        f"model exploded (observed at step {self._current_step})."
                    )

            # Write outputs
            self._write_losses(interval_losses, out_root)

            # Checkpoint if due
            if self._current_step >= self._next_checkpoint_step:
                self._write_checkpoint(out_root)
                self._next_checkpoint_step += checkpoint_interval_steps

            # Check for convergence
            if convergence_tolerance is not None:

                if self._relative_change < convergence_tolerance:
                    self._patience_counter += 1
                else:
                    self._patience_counter = 0

                if self._patience_counter >= patience:
                    converged = True
                    self._write_checkpoint(out_root)
                    break

        params = {k: np.asarray(v.detach().cpu()) for k, v in pyro.get_param_store().items()}
        final_state = pyro.get_param_store().get_state()
        return final_state, params, converged

    def _get_genotype_dim_map(self):
        """
        Identify which sites are in the genotype plate and what that dimension is.

        Returns
        -------
        dict
            Dictionary mapping site names to their genotype dimension index.
        """

        torch.manual_seed(0)
        model_trace = poutine.trace(self.model.pyro_model).get_trace(
            data=self.model.data,
            priors=self.model.priors
        )

        total_num_genotypes = self.model.data.num_genotype

        dim_map = {}
        genotype_dim = -1  # default fallback

        # First pass: find a site with the genotype plate to identify the dim index
        for name, site in model_trace.nodes.items():
            for frame in site.get("cond_indep_stack", []):
                if "genotype" in frame.name.lower():
                    genotype_dim = frame.dim
                    break
            if genotype_dim != -1:
                break

        # Second pass: map all sites that are in the plate or match the genotype size
        for name, site in model_trace.nodes.items():
            if site["type"] not in ["sample", "deterministic"]:
                continue

            # Check plate stack first (most robust)
            in_plate = False
            for frame in site.get("cond_indep_stack", []):
                if "genotype" in frame.name.lower():
                    dim_map[name] = frame.dim
                    in_plate = True
                    break

            if in_plate:
                continue

            # Fallback for deterministics computed outside the plate
            # but matching the genotype size at the expected dimension.
            val = site["value"]
            if hasattr(val, "shape"):
                actual_dim = genotype_dim if genotype_dim >= 0 else len(val.shape) + genotype_dim
                if actual_dim >= 0 and actual_dim < len(val.shape):
                    if val.shape[actual_dim] == total_num_genotypes:
                        dim_map[name] = genotype_dim

        return dim_map

    def get_posteriors(self,
                       svi,
                       svi_state,
                       out_root,
                       num_posterior_samples=10000,
                       sampling_batch_size=100,
                       forward_batch_size=512):
        """
        Generate and save posterior samples using the trained guide.

        Uses `pyro.infer.Predictive` to sample from the posterior distribution
        defined by the guide (whose parameters are in the param store). Handles
        large datasets by batching predictions and writing to disk (HDF5).

        Parameters
        ----------
        svi : pyro.infer.SVI
            The SVI object used for inference.
        svi_state : dict or any
            The final param store state (or any value; params are read from
            the Pyro param store directly).
        out_root : str
            Root name for the output file.
        num_posterior_samples : int, optional
            Number of posterior samples to draw (default 10000).
        sampling_batch_size : int, optional
            Batch size for generating posterior samples (default 100).
        forward_batch_size : int, optional
            Batch size for calculating forward predictions (default 512).
        """

        guide = svi.guide

        # Get the mapping of site names to genotype dimension
        dim_map = self._get_genotype_dim_map()

        total_num_genotypes = self.model.data.num_genotype

        # Adjust sampling_batch_size if smaller than num_posterior_samples
        sampling_batch_size = min(sampling_batch_size, num_posterior_samples)
        num_latent_batches = -(-num_posterior_samples // sampling_batch_size)

        # Create a full-batch data object for sampling latents
        all_indices = torch.arange(total_num_genotypes, dtype=torch.long)
        full_data = self.model.get_batch(self.model.data, all_indices)

        latent_sampler = Predictive(guide, num_samples=sampling_batch_size)

        # Prepare HDF5 file
        h5_file = f"{out_root}_posterior.h5"

        samples_written = 0
        with h5py.File(h5_file, 'w') as hf:

            for batch_i in tqdm(range(num_latent_batches), desc="sampling posterior"):

                # Sample the guide posterior
                latent_samples = latent_sampler(priors=self.model.priors,
                                                data=full_data)

                # Sample batches of genotypes
                batch_collector = {}
                for start_idx in range(0, total_num_genotypes, forward_batch_size):

                    end_idx = min(start_idx + forward_batch_size, total_num_genotypes)
                    batch_indices = torch.arange(start_idx, end_idx, dtype=torch.long)

                    # Slice latent samples using the dim_map
                    batch_latents = {}
                    for k, v in latent_samples.items():
                        if k in dim_map:
                            actual_dim = dim_map[k] if dim_map[k] >= 0 else v.ndim + dim_map[k]
                            idx = torch.arange(start_idx, end_idx, dtype=torch.long)
                            batch_latents[k] = torch.index_select(v, actual_dim, idx)
                        else:
                            batch_latents[k] = v

                    # Get a batch of data
                    batch_data = self.model.get_batch(self.model.data, batch_indices)

                    # Forward pass for this batch
                    forward_sampler = Predictive(self.model.pyro_model,
                                                 posterior_samples=batch_latents)
                    batch_pred = forward_sampler(priors=self.model.priors,
                                                 data=batch_data)

                    # Collect samples (latents + predictions) for this batch
                    for k, v in batch_pred.items():
                        if k not in batch_collector:
                            batch_collector[k] = []
                        v_np = np.asarray(v.detach().cpu()) if isinstance(v, torch.Tensor) else np.asarray(v)
                        batch_collector[k].append(v_np)

                    # Latents not in predictions (e.g. guide-only parameters)
                    for k, v in batch_latents.items():
                        if k in batch_pred:
                            continue

                        if k not in batch_collector:
                            batch_collector[k] = []

                        v_np = np.asarray(v.detach().cpu()) if isinstance(v, torch.Tensor) else np.asarray(v)

                        # If global (not in dim_map), only add on first genotype batch
                        if k not in dim_map:
                            if start_idx == 0:
                                batch_collector[k].append(v_np)
                        else:
                            batch_collector[k].append(v_np)

                # Concatenate genotype batches for this latent batch
                this_batch_results = {}
                for k, v_list in batch_collector.items():
                    if k in dim_map:
                        # Concatenate along the genotype dimension
                        axis = dim_map[k]
                        this_batch_results[k] = np.concatenate(v_list, axis=axis)
                    else:
                        this_batch_results[k] = v_list[0]

                # Write to HDF5 file
                batch_size_actual = next(iter(this_batch_results.values())).shape[0]
                for k, v in this_batch_results.items():
                    if k not in hf:
                        maxshape = (num_posterior_samples,) + v.shape[1:]
                        chunks = (min(sampling_batch_size, 100),) + v.shape[1:]
                        hf.create_dataset(k, shape=maxshape, dtype=v.dtype, chunks=chunks)

                    hf[k][samples_written:samples_written + batch_size_actual] = v

                samples_written += batch_size_actual

            # Add metadata to HDF5 file
            hf.attrs["num_samples"] = samples_written
            hf.flush()

    def get_key(self):
        """
        Get a new random integer key for seeding purposes.

        Returns
        -------
        int
            A new unique integer key drawn from the internal RNG.
        """
        return int(self._rng.integers(0, 2**31))

    def _jitter_init_parameters(self, init_params, init_param_jitter):
        """
        Apply multiplicative log-normal jitter to initial parameter values.

        Parameters
        ----------
        init_params : dict
            Dictionary of initial parameter values (scalars or arrays).
        init_param_jitter : float
            Standard deviation of the log-normal noise. Set to 0 to disable.

        Returns
        -------
        dict
            Dictionary of jittered initial parameter values.
        """

        if init_param_jitter == 0:
            return init_params

        for p in init_params:
            v = init_params[p]

            if np.isscalar(v) or (hasattr(v, 'shape') and v.shape == ()):
                noise = float(self._rng.standard_normal())
            else:
                noise = self._rng.standard_normal(size=np.asarray(v).shape)

            init_params[p] = np.asarray(v) * np.exp(noise * init_param_jitter)

        return init_params

    def _write_checkpoint(self, out_root):
        """
        Atomically save the param store state and RNG state to a dill pickle file.

        Parameters
        ----------
        out_root : str
            Root name for the output checkpoint file.
        """

        ps_state = pyro.get_param_store().get_state()

        out_dict = {"rng_state": self._rng.bit_generator.state,
                    "svi_state": ps_state,
                    "current_step": self._current_step}

        tmp_checkpoint_file = f"{out_root}_checkpoint.tmp.pkl"
        checkpoint_file = f"{out_root}_checkpoint.pkl"

        with open(tmp_checkpoint_file, 'wb') as f:
            dill.dump(out_dict, f)
        os.replace(tmp_checkpoint_file, checkpoint_file)

    def _restore_checkpoint(self, checkpoint_file):
        """
        Load a param store state and RNG state from a checkpoint file.

        Parameters
        ----------
        checkpoint_file : str
            Path to the checkpoint .pkl file.

        Returns
        -------
        dict
            The restored param store state.
        """

        with open(checkpoint_file, "rb") as f:
            checkpoint_data = dill.load(f)

        ps_state = checkpoint_data['svi_state']

        if 'rng_state' in checkpoint_data:
            self._rng.bit_generator.state = checkpoint_data['rng_state']

        if 'current_step' in checkpoint_data:
            self._current_step = checkpoint_data['current_step']

        if not isinstance(ps_state, dict):
            raise ValueError(
                f"checkpoint_file {checkpoint_file} does not appear to have a saved svi_state"
            )

        return ps_state

    def _write_losses(self, losses, out_root):
        """
        Write losses to binary (.bin) and text (.txt) files.

        Parameters
        ----------
        losses : array-like
            Loss values from the recent optimization interval.
        out_root : str
            Root name for the output files.
        """

        losses_file = f"{out_root}_losses.bin"
        readable_losses_file = f"{out_root}_losses.txt"

        if self._current_step == 0:
            if os.path.exists(losses_file):
                os.remove(losses_file)
            if os.path.exists(readable_losses_file):
                os.remove(readable_losses_file)

        if len(losses) == 0:
            return

        with open(losses_file, "ab") as f:
            np.array(losses).tofile(f)
            f.flush()
            os.fsync(f.fileno())

        with open(readable_losses_file, "a") as f:
            f.write(f"{losses[-1]},{self._relative_change}\n")
            f.flush()
            os.fsync(f.fileno())

    def _update_loss_deque(self, losses):
        """
        Update the loss deque and calculate the convergence metric.

        The metric is: abs(mean(old_half) - mean(new_half)) / std(total_history)

        Parameters
        ----------
        losses : array-like
            New losses to add to the deque.
        """

        self._loss_deque.extend(list(losses))

        if len(self._loss_deque) < self._loss_deque.maxlen:
            self._relative_change = np.inf
            return

        history = np.array(self._loss_deque)
        half = len(history) // 2

        old_half = history[:half]
        new_half = history[half:]

        mean_old = np.mean(old_half)
        mean_new = np.mean(new_half)

        diffs = np.diff(history)
        std_history = np.std(diffs) / np.sqrt(2)

        if std_history < 1e-9:
            if np.isclose(mean_new, mean_old):
                self._relative_change = 0.0
            else:
                self._relative_change = np.inf
            return

        se = 2 * std_history / np.sqrt(self._loss_deque.maxlen)
        z_score = (mean_new - mean_old) / se

        self._relative_change = np.abs(z_score)

    def write_params(self, params, out_root):
        """
        Write parameters to an .npz file.

        Parameters
        ----------
        params : dict
            Dictionary of parameters.
        out_root : str
            String to prepend to the output filename.
        """

        tmp_out_file = f"{out_root}_params.tmp.npz"
        out_file = f"{out_root}_params.npz"

        np.savez_compressed(tmp_out_file, **params)
        os.replace(tmp_out_file, out_file)

    def _get_site_names(self, target_sites="deterministic"):
        """
        Dry-runs the Pyro model to extract site names.

        Parameters
        ----------
        target_sites : str or list of str
            The type of sites to extract (e.g., 'deterministic', 'sample').

        Returns
        -------
        list
            List of site names found in the model trace.
        """

        if isinstance(target_sites, str):
            target_sites = [target_sites]

        torch.manual_seed(0)
        model_trace = poutine.trace(self.model.pyro_model).get_trace(
            data=self.model.data,
            priors=self.model.priors
        )

        site_names = [
            name for name, site_info in model_trace.nodes.items()
            if site_info["type"] in target_sites
        ]

        return site_names
