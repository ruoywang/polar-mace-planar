###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter
from mace.modules.loss import (
    attach_density_3d_samples_to_batch,
    density_3d_residuals,
    potential_1d_profile_residuals,
    predict_potential_from_dipole,
    solvent_layer_mean_residuals,
)

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    charge_msg = ""
    if eval_metrics.get("rmse_q") is not None:
        charge_msg = f", RMSE_Q={eval_metrics['rmse_q']:.4f} e"
    potential_msg = ""
    if eval_metrics.get("rmse_potential") is not None:
        potential_msg = f", RMSE_potential={eval_metrics['rmse_potential']:.4f} eV"
    if eval_metrics.get("rmse_fermi_level") is not None:
        potential_msg += f", RMSE_fermi={eval_metrics['rmse_fermi_level']:.4f} eV"
    density_msg = ""
    if eval_metrics.get("rmse_density_3d") is not None:
        density_msg += (
            f", RMSE_density_3d={eval_metrics['rmse_density_3d']:.5f} e/A^3"
        )
    if eval_metrics.get("rmse_potential_1d_profile") is not None:
        density_msg += (
            f", RMSE_potential_1d_profile={eval_metrics['rmse_potential_1d_profile']:.5f} eV"
        )
    if eval_metrics.get("rmse_solvent_layer_mean") is not None:
        density_msg += (
            f", RMSE_solvent_layer_mean={eval_metrics['rmse_solvent_layer_mean']:.5f} A"
        )
    atomic_dipole_msg = ""
    if eval_metrics.get("rmse_atomic_dipole") is not None:
        atomic_dipole_msg = f", RMSE_atomic_dipole={eval_metrics['rmse_atomic_dipole']:.4f} eA"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye{charge_msg}{potential_msg}{density_msg}{atomic_dipole_msg}",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch:
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_metrics = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics.get(
                                    "rmse_e_per_atom"
                                ),
                                "valid_rmse_f": eval_metrics.get("rmse_f"),
                            }
                            if eval_metrics.get("rmse_q") is not None:
                                wandb_metrics["valid_rmse_q"] = eval_metrics["rmse_q"]
                            if eval_metrics.get("rmse_potential") is not None:
                                wandb_metrics["valid_rmse_potential"] = eval_metrics["rmse_potential"]
                            if eval_metrics.get("rmse_atomic_dipole") is not None:
                                wandb_metrics["valid_rmse_atomic_dipole"] = eval_metrics["rmse_atomic_dipole"]
                            if eval_metrics.get("mae_q") is not None:
                                wandb_metrics["valid_mae_q"] = eval_metrics["mae_q"]
                            if eval_metrics.get("mae_potential") is not None:
                                wandb_metrics["valid_mae_potential"] = eval_metrics["mae_potential"]
                            if eval_metrics.get("mae_atomic_dipole") is not None:
                                wandb_metrics["valid_mae_atomic_dipole"] = eval_metrics["mae_atomic_dipole"]
                            wandb_log_dict[valid_loader_name] = wandb_metrics
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb:
                wandb.log(wandb_log_dict)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        for batch in data_loader:
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    attach_density_3d_samples_to_batch(batch, loss_fn)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            attach_density_3d_samples_to_batch(batch, loss_fn)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    density_rng = getattr(loss_fn, "density_3d_rng", None)
    density_rng_state = None
    if density_rng is not None and hasattr(loss_fn, "density_3d_seed"):
        density_rng_state = density_rng.getstate()
        density_rng.seed(int(getattr(loss_fn, "density_3d_seed")))

    try:
        with preserve_grad_state(model):
            for batch in data_loader:
                batch = batch.to(device)
                attach_density_3d_samples_to_batch(batch, loss_fn)
                batch_dict = batch.to_dict()
                output = model(
                    batch_dict,
                    training=False,
                    compute_force=output_args["forces"],
                    compute_virials=output_args["virials"],
                    compute_stress=output_args["stress"],
                )
                avg_loss, aux = metrics(batch, output)
    finally:
        if density_rng is not None and density_rng_state is not None:
            density_rng.setstate(density_rng_state)
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Qs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("qs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_qs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Potentials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_potentials", default=[], dist_reduce_fx="cat")
        self.add_state(
            "FermiLevels_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_fermi_levels", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Density3D_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_density_3d", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Potential1DProfile_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_potential_1d_profile", default=[], dist_reduce_fx="cat")
        self.add_state(
            "SolventLayerMean_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_solvent_layer_mean", default=[], dist_reduce_fx="cat")
        self.add_state(
            "AtomicMus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_atomic_mus", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("charges") is not None and batch.charges is not None:
            self.qs.append(batch.charges)
            self.delta_qs.append(batch.charges - output["charges"])
            self.Qs_computed += filter_nonzero_weight(
                batch,
                self.delta_qs,
                batch.weight,
                batch.charges_weight,
                spread_atoms=True,
                spread_quantity_vector=False,
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if output.get("atomic_dipole") is not None and getattr(batch, "atomic_dipole", None) is not None:
            self.delta_atomic_mus.append(batch.atomic_dipole - output["atomic_dipole"])
            self.AtomicMus_computed += filter_nonzero_weight(
                batch,
                self.delta_atomic_mus,
                batch.weight,
                batch.atomic_dipole_weight,
                spread_atoms=True,
                spread_quantity_vector=True,
            )
        if output.get("potential") is None and output.get("dipole") is not None:
            pred_potential = predict_potential_from_dipole(
                ref=batch,
                pred=output,
                axis=int(getattr(self.loss_fn, "potential_axis", 2)),
                potential_sign=float(getattr(self.loss_fn, "potential_sign", 1.0)),
            )
            if pred_potential is not None:
                output["potential"] = pred_potential
        if output.get("potential") is not None and getattr(batch, "potential", None) is not None:
            self.delta_potentials.append(batch.potential.view(-1) - output["potential"].view(-1))
            self.Potentials_computed += filter_nonzero_weight(
                batch,
                self.delta_potentials,
                batch.weight,
                batch.potential_weight,
                spread_quantity_vector=False,
            )
        if (
            getattr(self.loss_fn, "fermi_level_weight", 0.0) > 1.0e-12
            and output.get("fermi_level_pred") is not None
            and getattr(batch, "fermi_level", None) is not None
        ):
            self.delta_fermi_levels.append(
                batch.fermi_level.view(-1) - output["fermi_level_pred"].view(-1)
            )
            self.FermiLevels_computed += torch.tensor(
                float(output["fermi_level_pred"].view(-1).numel()),
                dtype=self.FermiLevels_computed.dtype,
                device=self.FermiLevels_computed.device,
            )
        density_3d_targets = getattr(self.loss_fn, "density_3d_targets", None)
        if density_3d_targets:
            density_3d_res = density_3d_residuals(
                ref=batch,
                pred=output,
                density_targets=density_3d_targets,
                density_smearing_width=getattr(self.loss_fn, "density_3d_sigma", 0.5),
                samples_per_graph=int(getattr(self.loss_fn, "density_3d_samples", 0)),
                rng=getattr(self.loss_fn, "density_3d_rng", None),
            )
            if density_3d_res is not None:
                self.delta_density_3d.append(density_3d_res.detach())
                self.Density3D_computed += torch.tensor(
                    float(density_3d_res.numel()),
                    dtype=self.Density3D_computed.dtype,
                    device=self.Density3D_computed.device,
                )
        potential_1d_targets = getattr(self.loss_fn, "potential_1d_profile_targets", None)
        if potential_1d_targets:
            potential_1d_res = potential_1d_profile_residuals(
                ref=batch,
                pred=output,
                potential_targets=potential_1d_targets,
                density_smearing_width=getattr(self.loss_fn, "density_3d_sigma", 0.5),
                axis=int(getattr(self.loss_fn, "potential_axis", 2)),
                solvent_sigma_g=float(getattr(self.loss_fn, "solvent_sigma_g", 0.85)),
                align=str(getattr(self.loss_fn, "potential_1d_profile_align", "mean")),
                use_solvent_profile=bool(
                    getattr(self.loss_fn, "potential_1d_profile_use_solvent_profile", False)
                ),
            )
            if potential_1d_res is not None:
                self.delta_potential_1d_profile.append(potential_1d_res.detach())
                self.Potential1DProfile_computed += torch.tensor(
                    float(potential_1d_res.numel()),
                    dtype=self.Potential1DProfile_computed.dtype,
                    device=self.Potential1DProfile_computed.device,
                )
        if getattr(self.loss_fn, "solvent_center_weight", 0.0) > 1.0e-12:
            solvent_layer_res, pred_layer_mean, target_layer_mean = (
                solvent_layer_mean_residuals(
                    ref=batch,
                    pred=output,
                    potential_targets=getattr(self.loss_fn, "potential_1d_profile_targets", {}),
                    density_targets=getattr(self.loss_fn, "density_3d_targets", {}),
                    use_density_center=bool(
                        getattr(self.loss_fn, "use_density_center_target", False)
                    ),
                    use_partition_center=bool(
                        getattr(self.loss_fn, "use_partition_center_target", False)
                    ),
                    axis=int(getattr(self.loss_fn, "potential_axis", 2)),
                    potential_sign=float(getattr(self.loss_fn, "potential_sign", 1.0)),
                    sigma_g=float(getattr(self.loss_fn, "solvent_sigma_g", 0.85)),
                )
            )
            if solvent_layer_res is not None and pred_layer_mean is not None:
                output["solvent_layer_mean"] = pred_layer_mean
                self.delta_solvent_layer_mean.append(
                    solvent_layer_res.detach()
                )
                self.SolventLayerMean_computed += torch.tensor(
                    float(target_layer_mean.numel()),
                    dtype=self.SolventLayerMean_computed.dtype,
                    device=self.SolventLayerMean_computed.device,
                )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Qs_computed:
            qs = self.convert(self.qs)
            delta_qs = self.convert(self.delta_qs)
            aux["mae_q"] = compute_mae(delta_qs)
            aux["rel_mae_q"] = compute_rel_mae(delta_qs, qs)
            aux["rmse_q"] = compute_rmse(delta_qs)
            aux["rel_rmse_q"] = compute_rel_rmse(delta_qs, qs)
            aux["q95_q"] = compute_q95(delta_qs)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.Potentials_computed:
            delta_potentials = self.convert(self.delta_potentials)
            aux["mae_potential"] = compute_mae(delta_potentials)
            aux["rmse_potential"] = compute_rmse(delta_potentials)
            aux["q95_potential"] = compute_q95(delta_potentials)
        if self.FermiLevels_computed:
            delta_fermi_levels = self.convert(self.delta_fermi_levels)
            aux["mae_fermi_level"] = compute_mae(delta_fermi_levels)
            aux["rmse_fermi_level"] = compute_rmse(delta_fermi_levels)
            aux["q95_fermi_level"] = compute_q95(delta_fermi_levels)
        if self.Density3D_computed:
            delta_density_3d = self.convert(self.delta_density_3d)
            aux["mae_density_3d"] = compute_mae(delta_density_3d)
            aux["rmse_density_3d"] = compute_rmse(delta_density_3d)
            aux["q95_density_3d"] = compute_q95(delta_density_3d)
        if self.Potential1DProfile_computed:
            delta_potential_1d_profile = self.convert(self.delta_potential_1d_profile)
            aux["mae_potential_1d_profile"] = compute_mae(delta_potential_1d_profile)
            aux["rmse_potential_1d_profile"] = compute_rmse(delta_potential_1d_profile)
            aux["q95_potential_1d_profile"] = compute_q95(delta_potential_1d_profile)
        if self.SolventLayerMean_computed:
            delta_solvent_layer_mean = self.convert(self.delta_solvent_layer_mean)
            aux["mae_solvent_layer_mean"] = compute_mae(delta_solvent_layer_mean)
            aux["rmse_solvent_layer_mean"] = compute_rmse(delta_solvent_layer_mean)
            aux["q95_solvent_layer_mean"] = compute_q95(delta_solvent_layer_mean)
        if self.AtomicMus_computed:
            delta_atomic_mus = self.convert(self.delta_atomic_mus)
            aux["mae_atomic_dipole"] = compute_mae(delta_atomic_mus)
            aux["rmse_atomic_dipole"] = compute_rmse(delta_atomic_mus)
            aux["q95_atomic_dipole"] = compute_q95(delta_atomic_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
