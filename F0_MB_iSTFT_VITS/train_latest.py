import os
import gc
from tqdm import tqdm

import torch
from torch.amp import autocast
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import commons
from pqmf import PQMF

from text.symbols import symbols
from models import SynthesizerTrn, MultiPeriodDiscriminator
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler

import others.safetensors as safe
from others.stdout_wrapper import SAFE_STDOUT

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  subband_stft_loss
)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65520'
  # n_gpus = 1

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
      logger = utils.get_logger(hps.model_dir)
      logger.info(hps)
      utils.check_git_hash(hps.model_dir)
      writer = SummaryWriter(log_dir=hps.model_dir)
      writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32, 300, 400, 500, 600, 700, 800, 900, 1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
      train_dataset,
      num_workers=1, # 8 -> 1
      shuffle=False,
      pin_memory=True,
      collate_fn=collate_fn,
      batch_sampler=train_sampler,
      persistent_workers=True, # False -> True
    )

    if rank == 0:
      eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
      eval_loader = DataLoader(
        eval_dataset,
        num_workers=0,
        shuffle=False,
        batch_size=1, # hps.train.batch_size -> 1
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
      )

    # model
    net_g = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    # optimizer
    optim_g = torch.optim.AdamW(
      filter(lambda p: p.requires_grad, net_g.parameters()), # net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps
    )
    
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True) # Linux Only
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True) # Linux Only

    # _ = safe.load_safetensors(os.path.join(hps.model_dir, "G_0.safetensors"), net_g)
    # _ = safe.load_safetensors(os.path.join(hps.model_dir, "D_0.safetensors"), net_d)
    # logger.info("Loaded the pretrained models.")

    epoch_str = 1
    global_step = 0

    def lr_lambda(epoch):
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_g, lr_lambda=lr_lambda, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optim_d, lr_lambda=lr_lambda, last_epoch=epoch_str - 2)
    scaler = torch.amp.GradScaler(device="cuda", enabled=hps.train.bf16_run)
    
    logger.info("Start training.")

    diff = abs(epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader))
    pbar = tqdm(
        total=global_step + diff,
        initial=global_step,
        smoothing=0.05,
        file=SAFE_STDOUT,
        dynamic_ncols=True,
    )
    initial_step = global_step

    for epoch in range(epoch_str, hps.train.epochs + 1):
      if rank==0:
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, eval_loader],
            logger,
            [writer, writer_eval],
            pbar,
            initial_step
        )
      else:
        train_and_evaluate(
            rank,
            epoch,
            hps,
            [net_g, net_d],
            [optim_g, optim_d],
            [scheduler_g, scheduler_d],
            scaler,
            [train_loader, None],
            None,
            None,
            pbar,
            initial_step
        )

      if epoch == hps.train.epochs:
          # Save the final models
          assert optim_g is not None
          utils.save_checkpoint(
              net_g,
              optim_g,
              hps.train.learning_rate,
              epoch,
              os.path.join(hps.model_dir, f"G_{global_step}.pth"),
          )
          assert optim_d is not None
          utils.save_checkpoint(
              net_d,
              optim_d,
              hps.train.learning_rate,
              epoch,
              os.path.join(hps.model_dir, f"D_{global_step}.pth"),
          )
          utils.save_checkpoint(
              net_g,
              epoch,
              f"{hps.model_name}_e{epoch}_s{global_step}.safetensors",
              for_infer=True,
          )
    pbar.close()


def compute_subband_loss(hps, y, y_hat_mb):
    pqmf = PQMF(y.device)
    y_mb = pqmf.analysis(y)
    return subband_stft_loss(hps, y_mb, y_hat_mb)


def compute_kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    return kl_loss(z_p, logs_q, m_p, logs_p, z_mask)


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    pbar: tqdm,
    initial_step: int,
):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders

  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, f0_gt) in enumerate(train_loader):
    f0_gt = f0_gt.cuda(rank, non_blocking=True)
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
      y_hat, y_hat_mb, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q), f0_pred = net_g(x, x_lengths, spec, spec_lengths)

      f0_pred = torch.nan_to_num(f0_pred, nan=0.0, posinf=0.0, neginf=0.0)

      # mel
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      y_mel = commons.slice_segments(
          mel,
          ids_slice,
          hps.train.segment_size // hps.data.hop_length
      )
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      y = commons.slice_segments(
          y,
          ids_slice * hps.data.hop_length,
          hps.train.segment_size
      )

      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

      with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

    # loss_gen_all...の下に移動
    # optim_d.zero_grad()

    # scaler.scale(loss_disc_all).backward()
    # scaler.unscale_(optim_d)
    # scaler.step(optim_d)   

    with autocast("cuda", enabled=hps.train.bf16_run, dtype=torch.bfloat16):
      # T = min(y_mel.size(2), y_hat_mel.size(2))
      # y_hat_mel[:, :, :T] y_mel[:, :, :T]
      loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
      loss_dur = torch.sum(l_length.float())
      loss_kl = compute_kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

      # loss
      if global_step < 5000:
        loss_fm = torch.tensor(0.0, device=y.device)
        loss_gen = torch.tensor(0.0, device=y.device)
        losses_gen = []
      else:
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
      
      # subband
      if hps.model.mb_istft_vits == True:
        loss_subband = compute_subband_loss(hps, y, y_hat_mb)
      else:
        loss_subband = torch.tensor(0.0)

      T = min(f0_pred.size(2), f0_gt.size(2))
      loss_f0 = F.l1_loss(f0_pred[:, :, :T] * z_mask[:, :, :T], f0_gt[:, :, :T] * z_mask[:, :, :T]) * hps.train.c_f0

      # loss
      if global_step < hps.train.warmup:
          loss_gen_all = loss_mel + loss_f0
          loss_disc_all = torch.tensor(0.0, device=y.device)

      elif global_step < hps.train.dur_step:
          loss_gen_all = loss_mel + loss_f0 + loss_dur
          loss_disc_all = loss_disc

      elif global_step < hps.train.kl_step:
          loss_gen_all = loss_mel + loss_f0 + loss_dur + loss_kl
          loss_disc_all = loss_disc

      else:
          loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband + loss_f0
          loss_disc_all = loss_disc

    # optim_d steps
    did_step_d = False

    if loss_disc_all is not None and loss_disc_all != 0:
      optim_d.zero_grad(set_to_none=True)
      scaler.scale(loss_disc_all).backward()
      scaler.unscale_(optim_d)
      # if getattr(hps.train, "bf16_run", False):
      #    torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
      scaler.step(optim_d)  
      did_step_d = True 

    # optim_g steps　
    did_step_g = False

    if loss_gen_all is not None and loss_gen_all != 0:
      optim_g.zero_grad(set_to_none=True)
      scaler.scale(loss_gen_all).backward()
      scaler.unscale_(optim_g)
      # if getattr(hps.train, "bf16_run", False):
      #    torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
      scaler.step(optim_g)
      scaler.update()
      did_step_g = True

    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/subband": loss_subband})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        utils.summarize(writer=writer, global_step=global_step, scalars=scalar_dict)
 
      if (
          global_step % hps.train.eval_interval == 0
          and global_step != 0
          and initial_step != global_step
        ):
        evaluate(hps, net_g, eval_loader, writer_eval)

        assert hps.model_dir is not None
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        safe.save_safetensors(net_g, epoch, os.path.join(hps.model_dir, f"{hps.model_name}_e{epoch}_s{global_step}.safetensors",), for_infer=True,)
    
    global_step += 1
    if pbar is not None:
        pbar.set_description(f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}")
        pbar.update()

  gc.collect()
  torch.cuda.empty_cache()

  if did_step_g:
    scheduler_g.step()
  if did_step_d:
    scheduler_d.step()

    
def evaluate(hps, generator, eval_loader, writer_eval):
    "Style-Bert-VITS2/train_ms.py"
    generator.eval()
    image_dict = {}
    audio_dict = {}

    logger = utils.get_logger(hps.model_dir)
    logger.info("Evaluating ...")

    with torch.inference_mode():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, f0_gt) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        # x = x[:1]
        # x_lengths = x_lengths[:1]
        # spec = spec[:1]
        # spec_lengths = spec_lengths[:1]
        # y = y[:1]
        # y_lengths = y_lengths[:1]
        # break

        for use_sdp in [True, False]:
          y_hat, y_hat_mb, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
          y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

          audio_dict.update({f"gen/audio_{batch_idx}_{use_sdp}": y_hat[0, :, : y_hat_lengths[0]]})
          audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
  main()
