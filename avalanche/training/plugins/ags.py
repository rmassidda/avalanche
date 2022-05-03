from functools import partial
from tqdm.auto import tqdm
from typing import Dict, List, NamedTuple

from torch.utils.data import DataLoader
from torch.nn import Module
import numpy as np
import torch

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.base_sgd import BaseSGDTemplate


class HookableModule(NamedTuple):
    module_name: str
    ingoing_module: Module
    fan_in: int
    fan_out: int
    outgoing_modules: List[Module]
    activations_mod: Module
    activations_idx: int


class AGSPlugin(SupervisedPlugin):
    """
    Adaptive Group Sparse (AGS) Regularization.

    The technique identifies a set of important artificial
    neurons and then penalizes the change of their parameters.
    Furthermore, the approach reinitializes uninmportant
    nodes.

    The original paper also described the use of proximity
    gradient descent that, for the structure of Avalanche
    should be left to an AGS strategy.

    Technique introduced in:
    "Continual Learning with Node-Importance based
    Adaptive Group Sparse Regularization"
    by Jung et. al (2020).
    """

    def __init__(self,
                 targets: List[HookableModule],
                 unimportant_reg: float = 1.,
                 important_reg: float = 1.,
                 decay_factor: float = 0.9,
                 reinit: bool = True,
                 reinit_prob: float = 0.5,
                 epsilon: float = 1e-8,
                 verbose=False):
        """
        :param unimportant_reg
        :param important_reg
        :param decay_factor
        :param reinit
        :param reinit_prob
        :param epsilon
        :param verbose
        """

        # Init super class
        super().__init__()

        # Store the modules where to act
        self.targets = targets

        # Model parameters
        self.params: Dict[str, torch.Tensor] = self._get_params(targets)

        # Model importance
        self.importance: Dict[str, np.ndarray] = {
            module.module_name: np.zeros(module.fan_out)
            for module in self.targets
        }

        # Decay factor
        self.decay = decay_factor

        # Regularization parameters
        self.unimportant_reg = unimportant_reg
        self.important_reg = important_reg
        self.epsilon = epsilon

        # Progress bar
        self.verbose = verbose

    def _get_params(self, targets: List[HookableModule]) \
            -> Dict[str, torch.Tensor]:
        # Init dictionary
        params: Dict[str, torch.Tensor] = {}

        # Iterate over the modules
        for module in targets:
            if isinstance(module.ingoing_module, torch.nn.Linear):
                # Copy the parameters
                self.params[module.module_name] = module.ingoing_module \
                    .weight.detach().cpu()
            else:
                raise ValueError("Module not supported")

        return params

    def _get_importance(self, targets: List[HookableModule],
                        old_importance: Dict[str, np.ndarray],
                        decay: float, strategy: BaseSGDTemplate,
                        verbose: bool = False) -> Dict[str, np.ndarray]:
        # Check whether the method is called during a
        # valid training experience
        if not strategy.experience:
            raise ValueError("Current experience is not available")

        if strategy.experience.dataset is None:
            raise ValueError("Current dataset is not available")

        # Activations buffer
        activations: Dict[str, list[np.ndarray]] = {
            module.module_name: [] for module in targets
        }

        # Average activations
        average: Dict[str, np.ndarray] = {
            module.module_name: np.zeros((module.fan_out,))
            for module in targets
        }

        # Init importance
        importance: Dict[str, np.ndarray] = {
            module_name: np.zeros_like(old_importance[module_name])
            for module_name in old_importance.keys()
        }

        # Define the forward hook
        def forward_hook(module: Module, input: torch.Tensor,
                         output: torch.Tensor, module_name: str):

            # Extract the output
            out = output.detach().cpu().numpy()

            # Update the activations sum
            activations[module_name].append(out.sum(axis=0))

        # Attach the hooks
        hooks_handles = []
        for module in targets:

            # Fill the hook arguments
            hook = partial(forward_hook, module_name=module.module_name)

            # Attach the hook
            handle = module.activations_mod.register_forward_hook(hook)

            # Append the handle
            hooks_handles.append(handle)

        # Forward pass to analyze activations
        strategy.model.eval()
        dataloader = DataLoader(
            strategy.experience.dataset,
            batch_size=strategy.train_mb_size,)  # type: ignore

        # Progress bar
        if verbose:
            print("Computing importance")
            dataloader = tqdm(dataloader)

        # Init number of examples
        n_examples = 0

        # Iterate over the dataset
        for _, batch in enumerate(dataloader):
            # Get batch
            if len(batch) == 2 or len(batch) == 3:
                x, _, t = batch[0], batch[1], batch[-1]
            else:
                raise ValueError("Batch size is not valid")

            # Move batch to device
            x = x.to(strategy.device)

            # Forward pass
            strategy.model.zero_grad()
            _ = avalanche_forward(strategy.model, x, t)

            # Analyze the activations
            for module in targets:

                # Extract the activations
                out = activations[module.module_name][module.activations_idx]

                # Update the average
                average[module.module_name] *= n_examples
                average[module.module_name] += out
                average[module.module_name] /= (n_examples + x.shape[0])

            # Update the number of examples
            n_examples += x.shape[0]

        # Detach the hooks
        for handle in hooks_handles:
            handle.remove()

        # In AGS the importance is given by the previous value
        # and the average activations
        for module in targets:
            importance[module.module_name] = \
                decay * old_importance[module.module_name] \
                + average[module.module_name]

        return importance

    def before_backward(self, strategy: BaseSGDTemplate, **kwargs):
        # Check if the task is not the first
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        # Check if properties have been initialized
        if self.importance is None:
            raise ValueError("Importance is not available")
        if self.params is None:
            raise ValueError("Parameters are not available")
        if strategy.loss is None:
            raise ValueError("Loss is not available")

        # Initialize penalty terms
        uninmportant_acc = 0.
        important_acc = 0.

        # Get current parameters
        new_params = self._get_params(self.targets)

        # Accumulate the penalty terms
        for module in self.targets:
            # Iterate over the "neurons"
            # TODO: there should be a smart way to vectorize this
            for i in range(module.fan_out):
                # Check if the neuron is important
                if self.importance[module.module_name][i] > self.epsilon:
                    # Important node
                    important_acc += self.importance[module.module_name][i] * \
                        torch.norm(new_params[module.module_name][i]
                                   - self.params[module.module_name][i])
                else:
                    # Unimportant node
                    uninmportant_acc += torch.norm(
                        new_params[module.module_name][i])

        # Update loss
        strategy.loss += self.unimportant_reg * uninmportant_acc
        strategy.loss += self.important_reg * important_acc
        return strategy.loss

    # NOTE: I am initializing everything in the constructor
    #       do I need to redefine this method?
    # def before_training(self, strategy: BaseSGDTemplate, **kwargs):

    def after_training_exp(self, strategy: BaseSGDTemplate, **kwargs):
        # Update parameters
        self.params = self._get_params(self.targets)

        # Update importance
        self.importance = self._get_importance(
            self.targets, self.importance, self.decay, strategy,
            self.verbose)
