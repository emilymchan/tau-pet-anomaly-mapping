"""
Adapted from ci-ber/2024-miccai-bercea-thor/projects/thor/DDPMTrainer.py
Implementation of the abstract class "Trainer" from core module.
Includes training and validation routines
"""

from core.Trainer import Trainer
from time import time
import os
import wandb
import logging
import matplotlib.pyplot as plt
from net_utils.simplex_noise import generate_noise
from optim.losses.image_losses import *
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torch.utils.data import DataLoader


def print_mem(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params["val_interval"]

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        # print("train()")
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer

        epoch_losses = []

        # self.train_ds = DataLoader(
        #     self.train_ds.dataset,
        #     batch_size=2,
        #     num_workers=2,
        #     shuffle=False,
        #     drop_last=True,
        #     pin_memory=True,
        # )

        self.early_stop = False
        # to handle loss with mixed precision training
        scaler = GradScaler()

        print("Before epoch loop")
        for epoch in range(self.training_params["nr_epochs"]):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info(
                    "[Trainer::test]: ################ Finished training (early stopping) ################"
                )
                break
            start_time = time()
            batch_loss, count_images = 1.0, 0
            accumulation_steps = 2
            self.model.train()

            for i, data in enumerate(self.train_ds):
                # print(i)
                # print(self.model.training)
                # Input
                if (
                    len(self.training_params["input_size"]) == 3
                    and len(data.shape) == 5
                ):
                    images = data.to(self.device)
                else:
                    images = data[0].to(self.device)
                count_images += images.shape[0]
                transformed_images = (
                    self.transform(images) if self.transform is not None else images
                )
                # print("transformed_images [0, 1] scaling: ", torch.min(transformed_images), torch.max(transformed_images))

                # In this version of monai, they clamp everything to [-1, 1] so rescale
                images = images.detach()
                transformed_images = transformed_images.detach()
                images = (images * 2) - 1
                transformed_images = (transformed_images * 2) - 1

                # print_mem(tag="before zero_grad")

                # self.optimizer.zero_grad()

                # print_mem(tag="after zero_grad")

                # for mixed precision training
                with autocast(dtype=torch.bfloat16):
                    # print(self.model.train_scheduler.alphas_cumprod[:6])
                    # Create timesteps
                    timesteps = torch.randint(
                        0,
                        self.model.train_scheduler.num_train_timesteps,
                        (transformed_images.shape[0],),
                        device=images.device,
                    ).long()

                    # Generate random noise and noisy images
                    noise = generate_noise(
                        self.model.train_scheduler.noise_type,
                        images,
                        self.model.train_scheduler.num_train_timesteps,
                    ).detach()
                    # noise = torch.zeros_like(images)
                    # print("noise [0, 1] scaling: ", torch.min(noise), torch.max(noise))

                    # print(
                    #     "transformed_images [-1, 1] scaling: ",
                    #     torch.min(transformed_images),
                    #     torch.max(transformed_images),
                    # )

                    # print_mem(tag="Before model")
                    # Get model prediction
                    pred = self.model(
                        inputs=transformed_images, noise=noise, timesteps=timesteps
                    )
                    # print("Pred out of model: ", torch.min(pred), torch.max(pred))
                    # pred = (pred + 1) / 2
                    # print(
                    #     "Pred after [0, 1] scaling: ", torch.min(pred), torch.max(pred)
                    # )
                    transformed_images = (transformed_images + 1) / 2
                    images = ((images + 1) / 2).detach()

                    # print_mem(tag="After model")

                    if self.model.prediction_type == "sample":
                        target = transformed_images
                        pred = (pred + 1) / 2
                    else:
                        target = noise

                    loss = self.criterion_rec(pred.float(), target.float())
                    # print("image range:", target.min().item(), target.max().item())
                    # print("pred range:", pred.min().item(), pred.max().item())
                    # print("loss:", loss.item())

                # print_mem(tag="Before backward")
                loss = loss / accumulation_steps  # Divide loss to average it
                scaler.scale(loss).backward()
                # print_mem(tag="After backward")

                # Log gradients
                # for name, param in self.model.named_parameters():
                #     if 'out' in name:
                #         if param.grad is not None:
                #             print(f'{name} grad range: {param.grad.min().item()}, {param.grad.max().item()}')

                if (i + 1) % accumulation_steps == 0:
                    # Perform optimizer step after accumulating gradients
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                batch_loss += loss.item() * images.size(0)

                # break

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print(
                "Epoch: {} \tTraining Loss: {:.6f}, computed in {} seconds for {} samples".format(
                    epoch, epoch_loss, end_time - start_time, count_images
                )
            )
            wandb.log({"Train/Loss_": epoch_loss, "_epoch_": epoch})

            # Save latest model
            torch.save(
                {
                    "model_weights": self.model.state_dict(),
                    "optimizer_weights": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(self.client_path, "latest_model.pt"),
            )

            # Run validation
            if (epoch + 1) % self.val_interval == 0 and epoch > 0:
                self.test(
                    self.model.state_dict(),
                    self.val_ds,
                    "Val",
                    self.optimizer.state_dict(),
                    epoch,
                )

            # Plot training example
            if (epoch + 1) % 25 == 0 and epoch > 0:
                # Test inference on a training sample
                sample = transformed_images[:1]
                sample = (sample * 2) - 1

                with torch.no_grad():
                    sample_, _ = self.model.sample_from_image(sample, noise_level=500)

                sample = ((sample + 1) / 2).detach().cpu()[0].numpy()
                sample_ = ((sample_ + 1) / 2).detach().cpu()[0].numpy()
                sample = self.inverse_rescale(sample, self.training_params["rescale"])
                sample_ = self.inverse_rescale(sample_, self.training_params["rescale"])

                elements = [sample, sample_]
                diffp, axarr = plt.subplots(
                    1, 2, gridspec_kw={"wspace": 0, "hspace": 0}
                )
                diffp.set_size_inches(2 * 4, 4)
                slice_to_plot = int(72 / 2)
                for i in range(len(axarr)):
                    axarr[i].axis("off")
                    v_max = 5
                    c_map = "gray" if v_max == 1 else "jet"
                    vmin = -1 if v_max == 1 else 0
                    axarr[i].imshow(
                        elements[i][0, :, :, slice_to_plot],
                        vmin=vmin,
                        vmax=v_max,
                        cmap=c_map,
                    )

                wandb.log(
                    {
                        "Train"
                        + "/Training_example": [
                            wandb.Image(diffp, caption="Epoch_" + str(epoch))
                        ]
                    }
                )

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task="Val", opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        # print("test()")
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + "_loss_mse": 0,
            task + "_loss_ssim": 0,
            task + "_loss_psnr": 0,
        }
        test_total = 0
        criterion_ssim = ssim().to(self.device)
        criterion_psnr = psnr().to(self.device)

        with torch.no_grad():
            for data in test_data:
                x = data  # data[0]

                (b, c, h, w, d) = x.shape
                test_total += b
                x = x.to(self.device)

                # print("x [0, 1] scaling: ", torch.min(x), torch.max(x))
                x = (x * 2) - 1
                # print("x [-1, 1] scaling: ", torch.min(x), torch.max(x))
                x_, _ = self.test_model.sample_from_image(
                    x, noise_level=self.model.noise_level_recon
                )
                # print("x_ out of model: ", torch.min(x_), torch.max(x_))
                x = (x + 1) / 2
                x_ = (x_ + 1) / 2
                # print("x_ [0, 1] scaling: ", torch.min(x_), torch.max(x_))

                loss_mse = self.criterion_MSE(x_, x)
                loss_ssim = criterion_ssim(x_, x)
                loss_psnr = criterion_psnr(x_, x)

                metrics[task + "_loss_mse"] += loss_mse.item() * x.size(0)
                metrics[task + "_loss_ssim"] += loss_ssim.item() * x.size(0)
                metrics[task + "_loss_psnr"] += loss_psnr.item() * x.size(0)
                # break

        rec = x_.detach().cpu()[0].numpy()
        img = x.detach().cpu()[0].numpy()
        # print("img [0, 1]: ", np.min(img), np.max(img))
        # print("rec [0, 1]: ", np.min(rec), np.max(rec))

        # Rescale back to original image range
        rec = self.inverse_rescale(rec, self.training_params["rescale"])
        img = self.inverse_rescale(img, self.training_params["rescale"])
        # print("img [0, 5]: ", np.min(img), np.max(img))
        # print("rec [0, 5]: ", np.min(rec), np.max(rec))

        elements = [img, rec, (img - rec) ** 2]
        v_maxs = [3, 3, 3]
        diffp, axarr = plt.subplots(
            1, len(elements), gridspec_kw={"wspace": 0, "hspace": 0}
        )
        diffp.set_size_inches(len(elements) * 4, 4)
        slice_to_plot = int(d / 2)
        for i in range(len(axarr)):
            axarr[i].axis("off")
            v_max = v_maxs[i]
            c_map = "gray" if v_max == 1 else "jet"
            vmin = -1 if v_max == 1 else 0
            axarr[i].imshow(
                elements[i][0, :, :, slice_to_plot], vmin=vmin, vmax=v_max, cmap=c_map
            )

        wandb.log(
            {task + "/Example": [wandb.Image(diffp, caption="Epoch_" + str(epoch))]}
        )

        for metric_key in metrics.keys():
            metric_name = task + "/" + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, "_step_": epoch})
        wandb.log({"lr": self.optimizer.param_groups[0]["lr"], "_step_": epoch})
        epoch_val_loss = metrics[task + "_loss_mse"] / test_total
        if task == "Val":
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save(
                    {
                        "model_weights": model_weights,
                        "optimizer_weights": opt_weights,
                        "epoch": epoch,
                    },
                    os.path.join(self.client_path, "best_model.pt"),
                )
                print("Best model saved")
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)

    def inverse_rescale(self, img, scaling):
        [in_low, in_high, out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        img = (img - b) / m

        return img
