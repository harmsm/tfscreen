
import jax
from jax import random
from jax import numpy as jnp

import numpyro

from numpyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
    init_to_value
)
from numpyro.optim import ClippedAdam

from numpyro.handlers import (
    trace, 
    seed
)

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
            - `jax_model` (callable): The Numpyro model.
            - `data` (flax.struct.dataclass): Data object, expected to have `num_genotype`.
            - `priors` (flax.struct.dataclass): Data object holding model priors
            - `init_params` (dict): Initial parameter guesses.
            - `sample_batch` (callable): Function to sample a data batch.
        seed : int
            Random seed for JAX PRNG key generation.
        """
        
        required_attr = ["data",
                         "priors",
                         "jax_model",
                         "jax_model_guide",
                         "init_params"]
        for attr in required_attr:
            if not hasattr(model,attr):
                raise ValueError(f"`model` must have attribute {attr}")
        
        self.model = model
        self._seed = seed
        self._main_key = random.PRNGKey(self._seed)
        self._current_step = 0


    def setup_map(self,
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  elbo_num_particles=1):
        """
        Set up SVI for MAP estimation using an AutoDelta guide.
        
        Parameters
        ----------
        adam_step_size : float, optional
            Step size for the ClippedAdam optimizer.
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.

        Returns
        -------
        numpyro.infer.SVI
            An SVI object configured with an AutoDelta guide.
        """

        # Create optimizer, guide, and svi object 
        optimizer = ClippedAdam(step_size=adam_step_size,clip_norm=adam_clip_norm)

        svi = SVI(self.model.jax_model,
                  self.model.jax_model_guide,
                  optimizer,
                  loss=Trace_ELBO(num_particles=elbo_num_particles))
        
        return svi

    def setup_svi(self,
                  init_params=None,
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  elbo_num_particles=10,
                  init_param_jitter=0.1,
                  init_scale=0.01):
        """
        Set up SVI. 

        Parameters
        ----------
        init_params : dict, optional
            starting parameter values. 
        adam_step_size : float, optional
            Step size for the ClippedAdam optimizer.
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.
        init_param_jitter : float, optional
            amount of jitter to apply to each parameter guess (log-normal). 
            Is only applied if init_params is not None. Default = 0.1. 
        init_scale : float, optional
            scale to apply to the initial parameter values. Is only applied if
            init_params is not None. 

        Returns
        -------
        numpyro.infer.SVI
            An SVI object
        """

        guide_kwargs = {}
        if init_params is not None:
            jittered_params = self._jitter_init_parameters(init_params,init_param_jitter)
            guide_kwargs["init_loc_fn"] = init_to_value(values=jittered_params)
            guide_kwargs["init_scale"] = init_scale
            
        optimizer = ClippedAdam(
            step_size=adam_step_size,
            clip_norm=adam_clip_norm)
        
        # ---------

        from numpyro.infer.util import initialize_model

        from jax import random
        import jax.numpy as jnp

        # You need a PRNGKey
        debug_key = random.PRNGKey(8675309)

        # args and kwargs should be exactly what you pass to svi.update or the model
        # Assuming 'args' is your tuple of arguments and 'kwargs' is your dict
        # If you usually pass them unpacked to SVI, unpack them here.

        print("\n========== DEBUGGING PARAMETERS ==========")
        try:
            # This simulates the first setup step of SVI
            init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                debug_key, 
                self.model.jax_model, 
                guide=self.model.jax_model_guide,
                model_args=[], # (arg1, arg2, ...)
                model_kwargs={"data":self.model.data,
                              "priors":self.model.priors} 
            )

            found_int = False
            print(f"{'PARAMETER NAME':<30} | {'DTYPE':<10} | {'SHAPE'}")
            print("-" * 60)
            
            for name, val in init_params.items():
                dtype_str = str(val.dtype)
                print(f"{name:<30} | {dtype_str:<10} | {val.shape}")
                
                # Flag anything that looks like an integer
                if "int" in dtype_str:
                    found_int = True
                    print(f"   >>> 🚨 FOUND INT PARAMETER: {name}")

            print("-" * 60)
            if not found_int:
                print("✅ No integer parameters found in the params dict.")
            else:
                print("❌ CRITICAL: Integer parameters detected. SVI will crash on these.")

        except Exception as e:
            print(f"Crash during model initialization debug: {e}")
            # If it crashes here, the issue is inside the guide/model trace generation itself
            import traceback
            traceback.print_exc()

        print("==========================================\n")


        svi = SVI(self.model.jax_model,
                  self.model.jax_model_guide,
                  optimizer,
                  loss=Trace_ELBO(num_particles=elbo_num_particles))
        
        return svi

    def run_optimization(self,
                         svi,
                         svi_state=None,
                         init_params=None,
                         out_root="tfs",
                         convergence_tolerance=1e-5,
                         convergence_window=1000,
                         checkpoint_interval=1000,
                         num_steps=10000000,
                         init_param_jitter=0.1):
        """
        Run the SVI optimization loop.

        This method iterates, updates SVI state, calculates loss, checks for
        convergence, and writes checkpoints.

        Parameters
        ----------
        svi : numpyro.infer.SVI
            The SVI object (from `setup_svi` or `setup_map`).
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
            Number of steps to average over for convergence check.
        checkpoint_interval : int, optional
            Frequency (in steps) to write checkpoints and check convergence.
        num_steps : int, optional
            Total number of optimization steps to run.
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

        # Set up initialization and update functions (triggers jit)
        init_function = jax.jit(svi.init)
        update_function = jax.jit(svi.update)

        # Get the arguments to pass to the jax model
        jax_model_kwargs = {"priors":self.model.priors,
                            "data":self.model.data} 

        # Add jitter to the input parameters if they are specified
        if init_params is not None:
            init_params = self._jitter_init_parameters(init_params=init_params,
                                                       init_param_jitter=init_param_jitter)

        # Create initialization key
        init_key = self.get_key()

        initial_svi_state = init_function(init_key,
                                          init_params=init_params,
                                          **jax_model_kwargs)
            
        if svi_state is None:
            svi_state = initial_svi_state
        else:
            if os.path.isfile(str(svi_state)):
                svi_state = self._restore_checkpoint(svi_state)
            else:
                svi_state = svi_state
            
        # loss deque holds loss values for smoothing to check for convergence
        self._loss_deque = deque(maxlen=(convergence_window*2))
        converged = False
        
        # Loop over all steps
        losses = []
        for i in range(num_steps):

            # Update the loss function
            svi_state, loss = update_function(svi_state,**jax_model_kwargs) 
            losses.append(loss)

            if i % checkpoint_interval == 0:

                self._current_step += i

                # Update loss deque
                self._update_loss_deque(losses,convergence_window)

                # stdout
                print(f"Step: {i:10d}, Loss: {loss:10.5e}, Change: {self._relative_change:10.5e}",flush=True)

                # Check for explosion in parameters
                params = svi.get_params(svi_state)
                for k in params:
                    if np.any(np.isnan(params[k])):
                        raise RuntimeError(
                            f"model exploded (observed at step {i})."
                        )
                    
                # Write outputs
                self._write_checkpoint(svi_state,out_root)
                self._write_losses(losses,out_root) 

                # Check for convergence               
                if convergence_tolerance is not None: 
                    if self._relative_change < convergence_tolerance:
                        converged = True
                        break

                # Reset losses list
                losses = []

        # Write final checkpoint and losses
        self._write_checkpoint(svi_state,out_root)
        self._write_losses(losses,out_root) 

        # Get current parameters
        params = svi.get_params(svi_state)
        
        return svi_state, params, converged

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
                svi object being used for the inference
            svi_state : 
                current state of the svi object
            out_root : str
                Root name for the output .npz file.
            num_posterior_samples : int, optional
                Number of posterior samples to draw.
            sampling_batch_size : int, optional
                Generate posteriors in blocks of this size
            forward_batch_size : int, optional
                calculate forward predictions in batches of this size
            """

            guide = svi.guide
            params = svi.get_params(svi_state)

            combined_results = {}

            total_num_genotypes = self.model.data.num_genotype 
            num_latent_batches = -(-num_posterior_samples // sampling_batch_size)

            for _ in tqdm(range(num_latent_batches),desc="sampling posterior"):

                # Sample the entire guide posterior distribution 
                post_key = self.get_key()
                latent_sampler = Predictive(guide,
                                            params=params,
                                            num_samples=sampling_batch_size)
                latent_samples = latent_sampler(post_key)

                # Create model kwargs
                jax_model_kwargs = {"priors":self.model.priors,
                                    "data":self.model.data} 

                # Sample batches of genotypes
                batched_results = {}
                for start_idx in range(0, total_num_genotypes, forward_batch_size):
                    
                    # Create indexes for slicing a batch
                    end_idx = min(start_idx + forward_batch_size, total_num_genotypes)
                    batch_indices = jnp.arange(start_idx, end_idx)

                    # Slice the Latent Samples to match this batch
                    batch_latents = {}
                    for k, v in latent_samples.items():

                        # Heuristic: if last dim matches total genotypes, slice it.
                        # Otherwise, treat it as a global parameter (keep full).
                        is_genotype_param = (v.ndim > 1) and (v.shape[-1] == total_num_genotypes)
                        
                        if is_genotype_param:
                            batch_latents[k] = v[..., start_idx:end_idx]
                        else:
                            batch_latents[k] = v


                    # Create a version of the model that only sees these indices
                    self.model.define_deterministic_model(batch_indices)
                    
                    # Create a sampler that will predict outputs using the full
                    # model given those latent samples
                    forward_sampler = Predictive(self.model.jax_model_deterministic, 
                                                 posterior_samples=batch_latents)

                    # Run the forward pass for this batch of genotypes
                    sample_key = self.get_key()
                    batch_pred = forward_sampler(sample_key, **jax_model_kwargs)
                    batch_pred_cpu = jax.device_get(batch_pred)

                    # Grab the parameters we sampled in this batch
                    for k, v in batch_pred_cpu.items():
                        if k not in batched_results:
                            batched_results[k] = []
                        batched_results[k].append(v)
                
                    # Grab the latent parameters we sampled in this batch. If 
                    # the latent parameter has a genotype axis (last dimension
                    # is length total_num_genotypes) slice and append it. If 
                    # it does not have a genotype axis, just append it on the 
                    # first iteration.
                    for k, v in latent_samples.items():

                        if k in batch_pred_cpu:
                            continue

                        if k not in batched_results:
                            batched_results[k] = []
                            
                        is_genotype_param = (v.ndim > 1) and (v.shape[-1] == (end_idx - start_idx))

                        if is_genotype_param:
                            batched_results[k].append(v)
                        else:
                            if start_idx == 0:
                                batched_results[k].append(v)

                # Combine our batched results. The result of this will be a 
                # dictionary of lists. Each list will be an array whose first
                # dimension is the number of latent samples. 
                for k, v in batched_results.items():

                    if k not in combined_results:
                        combined_results[k] = []

                    if len(v) == 1:
                        combined_results[k].append(v[0])
                    else:
                        full_width = np.concatenate(v,axis=1)
                        combined_results[k].append(full_width)
        
            # assemble final results
            final_results = {}
            for k in combined_results:
                final_results[k] = np.concatenate(combined_results[k],axis=0)

            self._write_posteriors(final_results, out_root=out_root)

    def predict(self,
                posterior_samples,
                predict_sites=None,
                data_for_predict=None):
        """
        Use the model to predict values of deterministic parameters in 
        predict_sites given the model and the model parameter samples in the
        posteriors_file. 

        posterior_samples : dict or str
            posteriors dictionary or path to a file holding posterior samples
            (usually generated by self.get_posteriors()) 
        predict_sites : list of str or str
            list of sites defined in the model we should predict. If None, 
            get all 'deterministic' sites. 
        data_for_predict : flax.dataclass
            dataclass exactly matching self.model.data in its parameters but 
            with (possibly) different independent variables (e.g. t_sel, 
            titrant_conc, etc.). This lets us predict at more titrant
            concentrations than the training set, for example. If None, use the
            data class used for training. 
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

        # Delete an existing losses_file if it is here and we are just starting
        # the run. 
        if self._current_step == 0:
            if os.path.exists(losses_file):
                os.remove(losses_file)

        # No losses to write this iteration, continue
        if len(losses) == 0:
            return 

        # Open in append Binary mode
        with open(losses_file, "ab") as f:
            np.array(losses).tofile(f)
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


    def _update_loss_deque(self,losses,convergence_window):
        """
        Update the loss deque and calculate the relative change.
        
        Compares the mean of the first half of the deque to the mean
        of the second half. The deque's maxlen is `2 * convergence_window`.

        Parameters
        ----------
        losses : list
            List of new losses to add to the deque.
        convergence_window : int
            The size of the window for averaging.
        """

        # Update loss history
        self._loss_deque.extend(list(losses))

        # Check if the deque is full (i.e., we have enough history)
        if len(self._loss_deque) < self._loss_deque.maxlen:
            self._relative_change = np.inf
            return 

        # Split the history into the "previous" and "current" windows
        history = np.array(self._loss_deque)
        prev_window_losses = history[:convergence_window]
        curr_window_losses = history[convergence_window:]

        # Calculate the mean of each window
        mean_prev = np.mean(prev_window_losses)
        mean_curr = np.mean(curr_window_losses)

        # Calculate relative change
        self._relative_change = np.abs(mean_curr - mean_prev) / (np.abs(mean_prev) + 1e-10)

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
