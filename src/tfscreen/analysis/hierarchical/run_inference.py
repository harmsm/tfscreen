
import jax
from jax import random
from jax import numpy as jnp

import numpyro
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoLowRankMultivariateNormal
)
from numpyro.infer import (
    SVI,
    Trace_ELBO,
    Predictive,
    init_to_value
)
from numpyro.optim import ClippedAdam

import pandas as pd
import numpy as np
import dill

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
            - `jax_model_kwargs` (dict): Keyword arguments for the model.
            - `init_params` (dict): Initial parameter guesses.
            - `data` (flax.struct.dataclass): Data object, expected to have `num_genotype`.
            - `sample_batch` (callable): Function to sample a data batch.
            - `deterministic_batch` (callable): Function for deterministic batch.
        seed : int
            Random seed for JAX PRNG key generation.
        """
        
        required_attr = ["jax_model_kwargs",
                         "jax_model",
                         "init_params",
                         "static_arg_names"]
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
        guide = AutoDelta(self.model.jax_model)

        svi = SVI(self.model.jax_model,
                  guide,
                  optimizer,
                  loss=Trace_ELBO(num_particles=elbo_num_particles))
        
        return svi

    def setup_svi(self,
                  init_params=None,
                  adam_step_size=1e-6,
                  adam_clip_norm=1.0,
                  guide_rank=10,
                  elbo_num_particles=10,
                  init_scale=1e-3):
        """
        Set up SVI with an AutoLowRankMultivariateNormal guide.

        Parameters
        ----------
        init_params : dict, optional
            starting parameter values. 
        adam_step_size : float, optional
            Step size for the ClippedAdam optimizer.
        adam_clip_norm : float, optional
            Gradient clipping norm for the ClippedAdam optimizer.
        guide_rank : int, optional
            Rank of the low-rank approximation for the guide's covariance.
        elbo_num_particles : int, optional
            Number of particles for ELBO estimation.
        init_scale : float, optional
            scale to apply to the initial parameter values

        Returns
        -------
        numpyro.infer.SVI
            An SVI object configured with an AutoLowRankMultivariateNormal guide.
        """

        guide_kwargs = {}
        if init_params is not None:
            guide_kwargs["init_loc_fn"] = init_to_value(values=init_params)
            guide_kwargs["init_scale"] = init_scale
            
        optimizer = ClippedAdam(step_size=adam_step_size,clip_norm=adam_clip_norm)
        guide = AutoLowRankMultivariateNormal(self.model.jax_model,
                                              rank=guide_rank,
                                              **guide_kwargs)
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
                         convergence_tolerance=1e-5,
                         convergence_window=1000,
                         checkpoint_interval=1000,
                         num_steps=10000000,
                         batch_size=1024):
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
        batch_size : int, optional
            Number of genotypes to include in each mini-batch.

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

        # Trim batch size if needed
        if batch_size > self.model.data.num_genotype:
            batch_size = self.model.data.num_genotype

        # Set up initialization and update functions (triggers jit)
        static_arg_names = self.model.static_arg_names
        init_function = jax.jit(svi.init, static_argnames=static_arg_names)
        update_function = jax.jit(svi.update, static_argnames=static_arg_names)

        # Get the arguments to pass to the jax model
        jax_model_kwargs = self.model.jax_model_kwargs.copy()
    
        # Create initialization key
        init_key = self.get_key()

        # Create dummy batch. Note that we just need the shape of the data, so 
        # we don't actually consume the key. 
        dummy_batch = self.model.sample_batch(init_key, self.model.data, batch_size)
        jax_model_kwargs["data"] = dummy_batch

        # Use the dummy batch and initial key to create the initial svi_state
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

            # Get key
            sample_key = self.get_key()

            # Create a mini-batch of the data
            batch_data = self.model.sample_batch(sample_key, self.model.data, batch_size)
            jax_model_kwargs["data"] = batch_data

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

    def get_parameter_posteriors(self,
                                 svi,
                                 svi_state,
                                 num_samples=10000,
                                 batch_size=100,
                                 out_dir="tfs_param-posteriors"):

        guide = svi.guide
        params = svi.get_params(svi_state)

        os.makedirs(out_dir, exist_ok=True)

        num_batches = int(np.ceil(num_samples / batch_size))
        for i in range(num_batches):

            # Sample batch on the GPU            
            batch_sampler = Predictive(guide,
                                       params=params,
                                       num_samples=batch_size)

            batch_key = self.get_key()
            batch_samples_gpu = batch_sampler(batch_key)
            
            # Pull down the samples and write to a file (atomic)
            batch_samples_cpu = jax.device_get(batch_samples_gpu)
            self._write_posteriors(out_root=f"{out_dir}/{i:05d}",
                                   posterior_samples=batch_samples_cpu)


    def get_posteriors(self,
                       svi,
                       svi_state,
                       out_root,
                       num_posterior_samples=10000,
                       batch_size=1024):
        """
        Generate and save posterior samples using the trained guide.

        Uses `numpyro.infer.Predictive` to sample from the posterior
        distribution defined by the guide and parameters. Handles large
        datasets by batching predictions.

        Parameters
        ----------

        out_root : str
            Root name for the output .npz file.
        num_posterior_samples : int, optional
            Number of posterior samples to draw.
        batch_size : int, optional
            Batch size for generating predictions. This does not need to
            match the training batch size.
        """

        guide = svi.guide
        params = svi.get_params(svi_state)

        # Sample the parameter posterior distribution
        post_key = self.get_key()
        latent_sampler = Predictive(guide,
                                    params=params,
                                    num_samples=num_posterior_samples)
        latent_samples = jax.device_get(latent_sampler(post_key))


        # Create a sampler that will predict outputs (obs) given those samples
        forward_sampler = Predictive(self.model.jax_model,
                                     posterior_samples=latent_samples)

        # Create model kwargs
        jax_model_kwargs = self.model.jax_model_kwargs.copy()

        # List to hold CPU-side results
        all_samples_cpu = []
        
        total_size = self.model.data.num_genotype 
        for start_idx in range(0, total_size, batch_size):
            
            # Create indexes for slicing a batch
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = jnp.arange(start_idx, end_idx)

            # Get a batch of data
            batch_data = self.model.deterministic_batch(self.model.data,
                                                        batch_indices)
            jax_model_kwargs["data"] = batch_data

            # Dispatch the forward pass for this batch.s
            sample_key = self.get_key()
            batch_samples_gpu = forward_sampler(sample_key, **jax_model_kwargs)

            # Retrieve results to CPU to free GPU memory for next batch.
            all_samples_cpu.append(jax.device_get(batch_samples_gpu))

        # Concatenate samples        
        final_samples = latent_samples.copy()
        forward_pass_keys = all_samples_cpu[0].keys()

        for key in forward_pass_keys:
            if key not in final_samples:
                
                # Collect all arrays for this key from all batches
                all_batches_for_key = [batch[key] for batch in all_samples_cpu]
                
                # Concatenate along the batch dimension (assumed to be -1)
                # This matches your original logic.
                final_samples[key] = np.concatenate(all_batches_for_key, axis=-1)

        self._write_posteriors(final_samples, out_root=out_root)



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
        Atomically write/append losses to a CSV file.

        Parameters
        ----------
        losses : list
            A list of loss values from the recent optimization interval.
        out_root : str
            Root name for the output losses CSV file.
        step_num : int
            number of the step this is from
        """

        # Name of output file
        losses_file = f"{out_root}_losses.csv"

        # Delete an existing losses_file if it is here and we are just starting
        # the run. 
        if self._current_step == 0:
            if os.path.exists(losses_file):
                os.remove(losses_file)

        # No losses to write this iteration, continue
        if len(losses) == 0:
            return 

        # Build dataframe of lossese
        losses_df = pd.DataFrame({"loss":np.array(losses,dtype=float)})

        # Append to existing if it's present
        if os.path.exists(losses_file):
            prev_df = pd.read_csv(losses_file)
            losses_df = pd.concat([prev_df,losses_df],ignore_index=True)

        # Atomic write
        tmp_loss_file = f"{out_root}_losses.tmp.csv"
        losses_df.to_csv(tmp_loss_file,index=False)
        os.replace(tmp_loss_file,losses_file)

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

