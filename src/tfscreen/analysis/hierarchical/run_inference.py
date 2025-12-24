
import jax
from jax import random
from jax import numpy as jnp

import numpyro
from numpyro.handlers import seed, trace

from numpyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
    init_to_value
)
from numpyro.optim import ClippedAdam
import optax

from numpyro.infer.autoguide import AutoDelta
import numpy as np
import dill
from tqdm.auto import tqdm

from collections import deque
import os

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
        self._relative_change = np.inf
        
        # Calculate iterations per epoch
        num_genotypes = self.model.data.num_genotype
        
        # Determine batch size by dry-running get_random_idx
        init_batch_key = int(self.get_key()[1])
        test_idx = self.model.get_random_idx(init_batch_key,num_batches=1)
        batch_size = len(test_idx.flatten())
        
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
        adam_step_size : float or callable, optional
            Step size for the ClippedAdam optimizer. Can be a fixed float or 
             a callable (e.g., an optax schedule).
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.
        guide_type : str, optional
            Type of guide to use. 
            - 'component' (default): Use the guide defined in the model components.
            - 'delta': Use numpyro.infer.autoguide.AutoDelta. Sets up a MAP estimator. 

        Returns
        -------
        numpyro.infer.SVI
            An SVI object
        """

        if guide_type == "delta":
            guide = numpyro.infer.autoguide.AutoDelta(self.model.jax_model)
        elif guide_type == "component":
            guide = self.model.jax_model_guide
        else:
            raise ValueError(f"guide_type '{guide_type}' not recognized.")

        optimizer = ClippedAdam(step_size=adam_step_size,
                                clip_norm=adam_clip_norm)   
        
        svi = SVI(self.model.jax_model,
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

        This method iterates, updates SVI state, calculates loss, checks for
        convergence, and writes checkpoints.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            The SVI object from `setup_svi`.
        svi_state : Any, optional
            An existing SVI state to resume from. If None, a new state is
            initialized. If a checkpoint file, restore from the checkpoint. 
        init_params : dict, optional
            Initial parameters. 
        out_root : str, optional
            Root name for output files (checkpoints, losses).
        convergence_tolerance : float, optional
            Relative change in smoothed loss to declare convergence.
        convergence_window : int, optional
            Number of epochs to average over for convergence check.
        patience : int, optional
            Number of consecutive convergence checks that must meet tolerance
            to declare convergence.
        convergence_check_interval : int, optional
            Frequency (in epochs) to check for convergence.
        checkpoint_interval : int, optional
            Frequency (in epochs) to write checkpoints.
        max_num_epochs : int, optional
            Maximum number of optimization epochs to run.
        init_param_jitter : float, optional
            amount of jitter to add to init_params. To turn off, set to 0.

        Returns
        -------
        svi_state : Any
            The final SVI state.
        params : dict
            The final optimized parameters.
        converged : bool
            True if the optimization converged based on the tolerance.
        
        Raises
        ------
        RuntimeError
            If parameters explode to NaN during optimization.
        """

        # Set up update function (triggers jit)
        update_function = jax.jit(svi.update)

        # Add jitter to the input parameters if they are specified
        if init_params is not None:
            init_params = self._jitter_init_parameters(init_params=init_params,
                                                       init_param_jitter=init_param_jitter)

        # Put the data on to the gpu
        data_on_gpu = jax.device_put(self.model.data)

        # JAX-optimized update function for use with lax.scan
        def scan_fn(carry, indices):
            svi_state = carry
            batch = self.model.get_batch(data_on_gpu, indices)
            new_svi_state, loss = update_function(svi_state,
                                                  priors=self.model.priors,
                                                  data=batch)
            return new_svi_state, loss

        # JIT the scan function
        fast_scan = jax.jit(lambda state, indices: jax.lax.scan(scan_fn, state, indices))

        # Create an initial batch to initialize SVI
        gpu_batch_idx = jax.device_put(self.model.get_random_idx())
        batch_data = self.model.get_batch(data_on_gpu, gpu_batch_idx)

        # Initialize svi with a batch of data
        init_key = self.get_key()
        initial_svi_state = svi.init(init_key,
                                     init_params=init_params,
                                     priors=self.model.priors,
                                     data=batch_data)
            
        if svi_state is None:
            svi_state = initial_svi_state
            
        elif isinstance(svi_state,str):
            if os.path.isfile(svi_state):
                svi_state = self._restore_checkpoint(svi_state)
            else:
                raise ValueError(
                    f"svi_state '{svi_state}' is not valid"
                )
        else:
            svi_state = svi_state
            
        # loss deque holds loss values for smoothing to check for convergence
        # The window represents epochs, so we multiply by iterations_per_epoch.
        # We need two such windows (one old, one new) to compare.
        deque_maxlen = 2 * convergence_window * self._iterations_per_epoch
        self._loss_deque = deque(maxlen=deque_maxlen)
        converged = False
        
        # Convert intervals from epochs to iterations
        check_interval_steps = convergence_check_interval * self._iterations_per_epoch
        checkpoint_interval_steps = checkpoint_interval * self._iterations_per_epoch
        total_steps = max_num_epochs * self._iterations_per_epoch
        
        # Track next checkpoint in steps
        # If resuming, we want to write checkpoint at the next multiple of 
        # checkpoint_interval_steps.
        self._next_checkpoint_step = ((self._current_step // checkpoint_interval_steps) + 1) * checkpoint_interval_steps
        
        # Reset patience counter
        self._patience_counter = 0
        
        # Loop over steps in chunks of check_interval_steps
        current_optimization_step = 0
        while current_optimization_step < total_steps:

            # Determine size of this block
            block_size = min(check_interval_steps, total_steps - current_optimization_step)

            # Generate a block of random indices (using NumPy/Python)
            block_idx = self.model.get_random_idx(num_batches=block_size)
            if block_idx.ndim == 1:
                block_idx = block_idx.reshape(1,-1)
            gpu_block_idx = jax.device_put(block_idx)

            # Run the block of updates using lax.scan (entirely on GPU)
            svi_state, block_losses = fast_scan(svi_state, gpu_block_idx)
            
            # Convert JAX array to NumPy for host-side metadata management
            # Ensure it is at least 1D for deque/IO
            interval_losses = np.atleast_1d(np.array(block_losses))
            
            # Update counters
            current_optimization_step += block_size
            self._current_step += block_size

            # Update loss deque for convergence check
            self._update_loss_deque(interval_losses)

            # stdout
            # Print status using the last loss in the block
            print(f"Step: {self._current_step:10d}, Loss: {interval_losses[-1]:10.5e}, Change: {self._relative_change:10.5e}, Patience: {self._patience_counter:3d}", flush=True)

            # Check for explosion in parameters
            params = svi.get_params(svi_state)
            for k in params:
                if np.any(np.isnan(params[k])):
                    raise RuntimeError(
                        f"model exploded (observed at step {self._current_step})."
                    )
                
            # Write outputs (checkpoints and losses)
            self._write_losses(interval_losses, out_root) 

            # Check if we should write a checkpoint
            if self._current_step >= self._next_checkpoint_step:
                self._write_checkpoint(svi_state, out_root)
                self._next_checkpoint_step += checkpoint_interval_steps

            # Check for convergence               
            if convergence_tolerance is not None: 
                
                # Formula: (mean(old) - mean(new)) / std(total)
                if self._relative_change < convergence_tolerance:
                    self._patience_counter += 1
                else:
                    self._patience_counter = 0

                if self._patience_counter >= patience:
                    converged = True
                    # Final checkpoint before exiting
                    self._write_checkpoint(svi_state, out_root)
                    break

        # Get final parameters
        params = svi.get_params(svi_state)
        
        return svi_state, params, converged

    def _get_genotype_dim_map(self):
        """
        Identify which sites are in the genotype plate and what that dimension is.

        Returns
        -------
        dict
            Dictionary mapping site names to their genotype dimension index.
        """

        # Run a trace of the model to identify plate structure
        seeded_model = seed(self.model.jax_model, rng_seed=0)
        traced_model = trace(seeded_model)
        model_trace = traced_model.get_trace(data=self.model.data,
                                             priors=self.model.priors)

        total_num_genotypes = self.model.data.num_genotype

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
                # Handle negative indexing for the dimension check
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

        Uses `numpyro.infer.Predictive` to sample from the posterior
        distribution defined by the guide and parameters. Handles large
        datasets by batching predictions.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            The SVI object being used for inference.
        svi_state : Any
            The current state of the SVI object (optimizer state).
        out_root : str
            Root name for the output .npz file.
        num_posterior_samples : int, optional
            Number of posterior samples to draw (default 10000).
        sampling_batch_size : int, optional
            Batch size for generating posterior samples of latent parameters
            (default 100).
        forward_batch_size : int, optional
            Batch size for calculating forward predictions (default 512).
        """

        guide = svi.guide
        params = svi.get_params(svi_state)

        # Get the mapping of site names to genotype dimension
        dim_map = self._get_genotype_dim_map()

        total_num_genotypes = self.model.data.num_genotype 

        # Adjust sampling_batch_size if smaller than num_posterior_samples
        sampling_batch_size = min(sampling_batch_size, num_posterior_samples)
        num_latent_batches = -(-num_posterior_samples // sampling_batch_size)

        # Create a full-batch data object for sampling latents
        all_indices = jnp.arange(total_num_genotypes)
        full_data = self.model.get_batch(self.model.data, all_indices)

        latent_sampler = Predictive(guide,
                                    params=params,
                                    num_samples=sampling_batch_size)

        combined_results = {}

        for _ in tqdm(range(num_latent_batches), desc="sampling posterior"):

            # Sample the guide posterior
            post_key = self.get_key()
            latent_samples = latent_sampler(post_key,
                                            priors=self.model.priors,
                                            data=full_data)

            # Sample batches of genotypes
            batch_collector = {}
            for start_idx in range(0, total_num_genotypes, forward_batch_size):
                
                end_idx = min(start_idx + forward_batch_size, total_num_genotypes)
                batch_indices = jnp.arange(start_idx, end_idx)

                # Slice latent samples using the dim_map
                batch_latents = {}
                for k, v in latent_samples.items():
                    if k in dim_map:
                        batch_latents[k] = jnp.take(v, jnp.arange(start_idx, end_idx), axis=dim_map[k])
                    else:
                        batch_latents[k] = v

                # Get a batch of data
                batch_data = self.model.get_batch(self.model.data, batch_indices)
                
                # Forward pass for this batch
                forward_sampler = Predictive(self.model.jax_model, 
                                             posterior_samples=batch_latents)
                sample_key = self.get_key()
                batch_pred = forward_sampler(sample_key,
                                             priors=self.model.priors,
                                             data=batch_data)

                # Collect all samples (latents + predictions) for this batch
                # Predictions from forward pass
                for k, v in batch_pred.items():
                    if k not in batch_collector:
                        batch_collector[k] = []
                    batch_collector[k].append(jax.device_get(v))
            
                # Latents that weren't in predictions (e.g. guide-only parameters)
                for k, v in batch_latents.items():
                    if k in batch_pred:
                        continue
                        
                    if k not in batch_collector:
                        batch_collector[k] = []
                    
                    # If global (not in dim_map), only add on first genotype batch
                    if k not in dim_map:
                        if start_idx == 0:
                            batch_collector[k].append(jax.device_get(v))
                    else:
                        # Geno-specific, always add
                        batch_collector[k].append(jax.device_get(v))

            # Concatenate genotype batches into the main results
            for k, v_list in batch_collector.items():
                if k not in combined_results:
                    combined_results[k] = []
                
                if k in dim_map:
                    # Concatenate along the genotype dimension (same as originally traced)
                    axis = dim_map[k]
                    combined_results[k].append(np.concatenate(v_list, axis=axis))
                else:
                    # Global parameter, just take the first (and only) entry
                    combined_results[k].append(v_list[0])
    
        # Final concatenation across latent batches
        final_results = {}
        for k, v_list in combined_results.items():
            final_results[k] = np.concatenate(v_list, axis=0)

        self._write_posteriors(final_results, out_root=out_root)


    def predict(self,
                posterior_samples,
                predict_sites=None,
                data_for_predict=None):
        """
        Use the model to predict values of deterministic sites.

        This method uses `numpyro.infer.Predictive` to generate predictions
        for specified sites, given posterior samples and potentially new
        input data.

        Parameters
        ----------
        posterior_samples : dict or str
            A dictionary of posterior samples or a path to a .npz file
            containing them (typically from `get_posteriors`).
        predict_sites : list of str or str, optional
            List of model sites to predict. If None, defaults to all
            'deterministic' sites found in the model trace.
        data_for_predict : object, optional
            A dataclass matching the structure of `self.model.data` but
            potentially containing different independent variables (e.g.,
            new time points or concentrations). If None, uses the original
            training data.

        Returns
        -------
        dict
            A dictionary mapping site names to predicted value arrays.
        """

        # Get all deterministic sites
        if predict_sites is None:
            predict_sites = self._get_site_names()

        # Make sure predict_sites is a list
        if isinstance(predict_sites,str):
            predict_sites = [predict_sites]

        # Load posterior samples from a file if necessary
        if isinstance(posterior_samples,str):
            posterior_samples = jnp.load(posterior_samples)
 
        # Convert to a dict if not a dict (for example, numpy.npz)
        if not isinstance(posterior_samples,dict):
            posterior_samples = dict(posterior_samples)

        # No data specified. Predict using the inputs
        if data_for_predict is None:
            data_for_predict = self.model.data

        # Set of predictor class
        predictor = Predictive(self.model.jax_model, 
                               posterior_samples=posterior_samples, 
                               return_sites=predict_sites)

        # Make predictions
        predict_key = self.get_key()
        predictions = predictor(predict_key, 
                                data=data_for_predict, 
                                priors=self.model.priors)
        
        return predictions

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


    def _write_checkpoint(self,svi_state,out_root):
        """
        Atomically save the SVI state and PRNG key to a dill pickle file.

        Parameters
        ----------
        svi_state : Any
            The SVI state (e.g., from `svi.update`).
        out_root : str
            Root name for the output checkpoint file.
        """

        host_svi_state = jax.device_get(svi_state)

        out_dict = {"main_key":self._main_key,
                    "svi_state":host_svi_state,
                    "current_step":self._current_step}

        tmp_checkpoint_file = f"{out_root}_checkpoint.tmp.pkl"

        checkpoint_file = f"{out_root}_checkpoint.pkl"

        # Atomic save
        with open(tmp_checkpoint_file,'wb') as f:
            dill.dump(out_dict,f)
        os.replace(tmp_checkpoint_file,
                   checkpoint_file)

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

        if not isinstance(svi_state,numpyro.infer.svi.SVIState):
            raise ValueError(
                f"checkpoint_file {checkpoint_file} does not appear to have a saved svi_state"
            )

        return svi_state

    def _write_losses(self,losses,out_root):
        """
        Write losses to a binary file. 

        Parameters
        ----------
        losses : list
            A list of loss values from the recent optimization interval.
        out_root : str
            Root name for the output losses CSV file.
        """

        # Name of output file
        losses_file = f"{out_root}_losses.bin"
        readable_losses_file = f"{out_root}_losses.txt"

        # Delete an existing losses_file if it is here and we are just starting
        # the run. 
        if self._current_step == 0:
            if os.path.exists(losses_file):
                os.remove(losses_file)
    
            if os.path.exists(readable_losses_file):
                os.remove(readable_losses_file)

        # No losses to write this iteration, continue
        if len(losses) == 0:
            return 

        # Open in append Binary mode
        with open(losses_file, "ab") as f:
            np.array(losses).tofile(f)
            f.flush()
            os.fsync(f.fileno())

        # Write a human-readable losses file
        with open(readable_losses_file,"a") as f:
            f.write(f"{losses[-1]},{self._relative_change}\n")
            f.flush()
            os.fsync(f.fileno())    


    def _write_posteriors(self,posterior_samples,out_root):
        """
        Atomically save posterior samples to a compressed .npz file.

        Parameters
        ----------
        posterior_samples : dict
            A dictionary of samples (pytree) from `numpyro.infer.Predictive`.
        out_root : str
            Root name for the output .npz file.
        """

        tmp_out_file = f"{out_root}_posterior.tmp.npz"
        out_file = f"{out_root}_posterior.npz"

        np.savez_compressed(tmp_out_file,**posterior_samples)

        os.replace(tmp_out_file,out_file)


    def _update_loss_deque(self, losses):
        """
        Update the loss deque and calculate the convergence metric.
        
        The metric is defined as:
        (mean(old_half) - mean(new_half)) / std(total_history)

        Parameters
        ----------
        losses : list
            List of new losses to add to the deque.
        """

        # Update loss history
        self._loss_deque.extend(list(losses))

        # Check if the deque is full (i.e., we have enough history to compare windows)
        if len(self._loss_deque) < self._loss_deque.maxlen:
            self._relative_change = np.inf
            return 

        # Split the history into the "previous" and "current" halves
        history = np.array(self._loss_deque)
        half = len(history) // 2
        
        old_half = history[:half]
        new_half = history[half:]

        # Calculate means and standard deviation
        mean_old = np.mean(old_half)
        mean_new = np.mean(new_half)
        std_history = np.std(history)

        # Calculate convergence metric
        # Use a small epsilon to avoid division by zero
        self._relative_change = (mean_old - mean_new) / (std_history + 1e-10)

    def write_params(self,params,out_root):
        """
        Write parameters to an .npz file.
        
        Parameters
        ----------
        params : dict
            dictionary of parameters
        out_root : str
            string to append to front of output file
        """

        tmp_out_file = f"{out_root}_params.tmp.npz"
        out_file = f"{out_root}_params.npz"

        np.savez_compressed(tmp_out_file,**params)

        os.replace(tmp_out_file,out_file)        

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
