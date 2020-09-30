import copy
import os
import torch
from tqdm.autonotebook import tqdm
import torch.optim as optim
import torch.nn as nn 
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from packaging import version

PYTORCH_VERSION = version.parse(torch.__version__)

try:
    from apex import amp
    IS_AMP_AVAILABLE = True
except ImportError:
    import logging

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.warning(
        "To enable mixed precision training, please install `apex`. "
        "Or you can re-install this package by the following command:\n"
        '  pip install torch-lr-finder -v --global-option="amp"'
    )

    IS_AMP_AVAILABLE = False
    del logging


class DataLoaderIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        if not isinstance(batch_data, list) and not isinstance(batch_data, tuple):
            raise ValueError("""Your batch type not supported: {}. Please inherit from `TrainDataLoaderIter`
                (or `ValDataLoaderIter`) and redefine
                `_batch_make_inputs_labels` method.""".format(type(batch_data)))

        inputs, labels, *_ = batch_data

        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self.inputs_labels_from_batch(batch)


class TrainDataLoaderIter(DataLoaderIter):

    def __init__(self, 
                data_loader, 
                auto_reset=True):

        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)

        return inputs, labels


class ValDataLoaderIter(DataLoaderIter):
    pass



class LRFinder(object):
        def __init__(self,
                    model, 
                    optimizer, 
                    criterion, 
                    device=None,
                    memory_cache = True, 
                    cache_dir = None):
            """
            Learning Rate Finder Class.
            Arguments:
              model: Base CNN Model.
              optimizer: Wrapped Optimizer.
              criterion: Loss Function implemented.  
              device: Assigned model on which the model computation takes place.    
            """
            
            self.optimizer = optimizer # check whether the optimizer is attached to the scheduler
            # self._check_for_scheduler()

            self.model = model
            self.criterion = criterion
            self.history = {'lr':[], 'loss':[]}
            self.bestLoss = None
            self.memory_cache = memory_cache

            # Save the original state of model & optimizer
            self.model_device = next(self.model.parameters()).device
            self.state_cacher = StateCacher(memory_cache, cache_dir= cache_dir)
            self.state_cacher.store('Model', self.model.state_dict())
            self.state_cacher.store('Optimizer', self.optimizer.state_dict())

            # If device is none then change it to default model_device
            if device:
                self.device = device
            else:
                self.device = self.model_device

        def reset(self):
            """ Restores Model & Optimizer to their initial states"""
            self.model.load_state_dict(self.state_cacher.retrieve('Model'))
            self.optimizer.load_state_dict(self.state_cacher.retrieve('Optimizer'))
            self.model.to(self.device) # Changed from self.model_device to self.device


        def range_test(
                    self,
                    train_loader,
                    val_loader=None,
                    start_lr=None, 
                    end_lr=10,
                    num_iter=100,
                    step_mode='linear', 
                    smooth_f=0.05, 
                    diverge_th=5,
                    accumulation_steps = 1,
                     non_blocking_transfer = True):

            self.history = {'lr':[], 'loss':[]}
            self.bestLoss = None

            # move the model to the device
            self.model.to(self.device)

            # check whether the optimizer is attached to the scheduler
            self._check_for_scheduler()

            # set the start of learning rate
            if start_lr:
                self._set_learning_rate(start_lr)

            # Initialize learning rate policy
            if step_mode.lower() == 'exp':
                lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
            elif step_mode.lower() == 'linear':
                lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
            else:
                raise ValueError('expected one of (exp, linear), got {a}'.format(a=step_mode))


            if smooth_f < 0 or smooth_f >= 1:
                raise ValueError('smooth_f is outside the range of [0,1]')


            # Create an iterator to get data batch by batch
            if isinstance(train_loader, DataLoader):
                train_iter = TrainDataLoaderIter(train_loader)
            elif isinstance(train_loader, TrainDataLoaderIter):
                train_iter = train_loader
            else:
                raise ValueError("""`train_loader` has unsupported type: {}.
                    Expected types are `torch.utils.data.DataLoader`
                    or child of `TrainDataLoaderIter`.""".format(type(train_loader)))

            if val_loader:
                if isinstance(val_loader, DataLoader):
                    val_iter = ValDataLoaderIter(val_loader)
                elif isinstance(val_loader, ValDataLoaderIter):
                    val_iter = val_loader
                else:
                    raise ValueError("""`val_loader` has unsupported type: {}.
                        Expected types are `torch.utils.data.DataLoader`
                        or child of `ValDataLoaderIter`.""".format(type(val_loader)))


            for iteration in tqdm(range(num_iter)):
                # Train on batch and retrieve loss
                # print(train_iter)
                loss = self._train_batch(train_iter, accumulation_steps, non_blocking_transfer=non_blocking_transfer)
                print("accumulation steps = ", accumulation_steps)
                if val_loader:
                    loss = self._validate(val_iter, non_blocking_transfer=non_blocking_transfer)

                # Update Learning Rate
                self.history['lr'].append(lr_schedule.get_lr()[0])
                lr_schedule.step()

                # Track the best loss
                if iteration == 0:
                    self.bestLoss = loss
                else:
                    if smooth_f>0:
                        loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                    if loss < self.bestLoss:
                        self.bestLoss = loss

                # check if the loss has diverged it needs to be stopped
                self.history['loss'].append(loss)
                if loss > diverge_th*self.bestLoss:
                    print("The loss has diverged, Stopping Early!")
                    break


        def _set_learning_rate(self, new_lrs):
            if not isinstance(new_lrs, list):
              new_lrs = [new_lrs] * len(self.optimizer.param_groups)
            if len(new_lrs) != len(self.optimizer.param_groups):
                  raise ValueError("Length of `new_lrs` is not equal to the number of parameter groups "+"in the given optimizer")

            for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
                param_group["lr"] = new_lr


        def _check_for_scheduler(self):
            for param_group in self.optimizer.param_groups:
                if "initial_lr" in param_group:
                    raise RuntimeError ("Optimizer has already a scheduler attached to it.")

        def _train_batch(self, 
                        train_iter,
                        accumulation_steps, 
                        non_blocking_transfer=True):


          self.model.train()
          total_loss = None  # for late initialization

          self.optimizer.zero_grad()
          for i in range(accumulation_steps):
            inputs, labels = train_iter.__next__()
            inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)


            # Forward pass
            outputs = self.model(inputs)
            print("outputs:", outputs)
            print("labels:", labels)
            loss = self.criterion(outputs, labels)
            print("loss = ", loss)
            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            if IS_AMP_AVAILABLE and hasattr(self.optimizer, "_amp_stash"):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

            self.optimizer.step()
            return total_loss.item()

        def _move_to_device(self, inputs, labels, non_blocking=True):
          def move(obj, device, non_blocking=True):
            if hasattr(obj, "to"):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking) for k, o in obj.items()}
            else:
                return obj

          inputs = move(inputs, self.device, non_blocking=non_blocking)
          labels = move(labels, self.device, non_blocking=non_blocking)

          return inputs, labels

        def _validate(self, val_iter, non_blocking_transfer=True):
          # Set model to evaluation mode and disable gradient computation
          running_loss = 0
          self.model.eval()

          with torch.no_grad():
              for inputs, labels in val_iter:
                  # Move data to the correct device
                  inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)

                  if isinstance(inputs, tuple) or isinstance(inputs, list):
                      batch_size = inputs[0].size(0)
                  else:
                      batch_size = inputs.size(0)

                  # Forward pass and loss computation
                  outputs = self.model(inputs)
                  loss = self.criterion(outputs, labels)
                  running_loss += loss.item() * batch_size

          return running_loss / len(val_iter.dataset)

        def plot(self, 
                skip_start=10, 
                skip_end=5, 
                log_lr=True,
                show_lr=None, 
                ax=None):
          """Plots the learning rate range test.
          Arguments:
              skip_start (int, optional): number of batches to trim from the start.
                  Default: 10.
              skip_end (int, optional): number of batches to trim from the start.
                  Default: 5.
              log_lr (bool, optional): True to plot the learning rate in a logarithmic
                  scale; otherwise, plotted in a linear scale. Default: True.
              show_lr (float, optional): if set, adds a vertical line to visualize the
                  specified learning rate. Default: None.
              ax (matplotlib.axes.Axes, optional): the plot is created in the specified
                  matplotlib axes object and the figure is not be shown. If `None`, then
                  the figure and axes object are created in this method and the figure is
                  shown . Default: None.
          Returns:
              The matplotlib.axes.Axes object that contains the plot.
          """

          if skip_start < 0:
              raise ValueError("skip_start cannot be negative")
          if skip_end < 0:
              raise ValueError("skip_end cannot be negative")
          if show_lr is not None and not isinstance(show_lr, float):
              raise ValueError("show_lr must be float")

          # Get the data to plot from the history dictionary. Also, handle skip_end=0
          # properly so the behaviour is the expected
          lrs = self.history["lr"]
          losses = self.history["loss"]

          if skip_end == 0:
              lrs = lrs[skip_start:]
              losses = losses[skip_start:]
          else:
              lrs = lrs[skip_start:-skip_end]
              losses = losses[skip_start:-skip_end]

          # Create the figure and axes object if axes was not already given
          fig = None
          if ax is None:
              fig, ax = plt.subplots()

          # Plot loss as a function of the learning rate
          ax.plot(lrs, losses)
          if log_lr:
              ax.set_xscale("log")

          ax.set_xlabel("Learning rate")
          ax.set_ylabel("Loss")

          if show_lr is not None:
              ax.axvline(x=show_lr, color="red")

          # Show only if the figure was created internally
          if fig is not None:
              plt.show()

          return ax


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    "Failed to load state in {}. File doesn't exist anymore.".format(fn)
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])


LR_List = []
Acc_List = []
def lr_rangetest(device, 
                model,
                trainloader, 
                criterion,  
                minlr, 
                maxlr, 
                epochs,
                weight_decay=0.05,
                plot=True):
    """
    Args:-
    """
    lr = minlr

    for e in range(epochs):
        testModel = copy.deepcopy(model)
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=weight_decay)
        lr = lr + (maxlr-minlr)/epochs
        testModel.train()
        pbar = tqdm(trainLoader)
        correct, processed = 0, 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = testModel(data)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
            processed = processed + len(data)
            pbar.set_description(desc= f'epoch = {e+1} Lr = {optimizer.param_groups[0]["lr"]}  Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        Acc_List.append(100*correct/processed)
        LR_List.append(optimizer.param_groups[0]['lr'])
    
    if plot:
        with plt.style.context('fivethirtyeight'):
            plt.plot(LR_List, Acc_List)
            plt.xlabel('Learning Rate')
            plt.ylabel('Accuracy')
            plt.title('Learning Rate Range Test')
            plt.show()