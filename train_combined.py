import torch
import utils.config as config
from tqdm.auto import tqdm
from utils.utils import Metrics
from utils.log import Logger

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, checkpoint_path, log_filename, device, resume=None, grad_accum=config.GRAD_ACCUM):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler  # ReduceLROnPlateau
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.log_filename = log_filename
        self.device = device
        self.grad_accum = grad_accum
        self.start_epoch = 0

        self.training_duration = 0
        self.patience_counter = 0
        self.best_weighted_metric = None
        self.best_val_loss = None
        self.best_val_f1 = None
        self.best_val_acc = None
        self.loss_weight = 0.3
        self.acc_weight = 0.0
        self.f1_weight = 0.7

        self.logger = Logger(log_filename)
        self.metric_calculator = Metrics(num_classes=config.NUM_CLASSES)
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_accs = []
        self.val_accs = []
        self.train_mious = []
        self.val_mious = []

        if resume:
            self._load_checkpoint(resume)

    def _load_checkpoint(self, resume):
        print(f"Resuming training...")
        checkpoint = torch.load(resume, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        #self.best_weighted_metric = checkpoint['val_loss']
        self.best_val_loss = checkpoint['val_loss']
        self.best_val_f1 = checkpoint['val_f1']
        self.best_val_acc = checkpoint['val_acc']

    def train(self):
        print('\n===========================')
        print('Start the training process:')

        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                train_loss, train_f1, train_acc = self._train_one_epoch(epoch)
                val_loss, val_f1, val_acc = self._validate(epoch)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_f1_scores.append(train_f1)
                self.val_f1_scores.append(val_f1)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)

                print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                      f'Train F1: {train_f1:.6f}, Val F1: {val_f1:.6f}, '
                      f'Train Acc: {train_acc:.6f}, Val Acc: {val_acc:.6f}')

                self.logger.log_epoch(epoch + 1, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc)

                # Update scheduler with validation loss
                self.scheduler.step()

                # Check early stopping
                self._check_early_stopping(val_loss, val_acc, val_f1, epoch)

                if self.patience_counter >= self.patience:
                    print('Early stopping due to no improvement in validation metrics')
                    break

        except Exception as e:
            print(f'Error during training: {e}')

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        f1_sum_train = 0.0
        acc_sum_train = 0.0
        num_samples = 0

        gradient_accumulation_steps = self.grad_accum
        accumulated_batches = 0

        training_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                            desc=f'Training Epoch {epoch + 1}/{self.num_epochs}')

        for step, (sentinel1, sentinel2, labels) in training_bar:
            sentinel1 = sentinel1.to(self.device, dtype=torch.float32)  
            sentinel2 = sentinel2.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)

            skip_batch = False
            with torch.cuda.amp.autocast():
                outputs = self.model(sentinel1, sentinel2)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    skip_batch = True
                else:
                    loss = self.criterion(outputs, labels)
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        skip_batch = True

            if skip_batch:
                print(f"[Warning] Skipping batch {step} due to NaN/Inf.")
                self.optimizer.zero_grad(set_to_none=True)
                continue  

            self.scaler.scale(loss).backward()
            accumulated_batches += 1

            if accumulated_batches % gradient_accumulation_steps == 0 or (step + 1 == len(self.train_loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            batch_size = sentinel1.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            f1_sum_train += self.metric_calculator.f1_score(outputs, labels).item() * batch_size
            acc_sum_train += self.metric_calculator.pixel_accuracy(outputs, labels).item() * batch_size

            training_bar.set_postfix({
                "loss": running_loss / num_samples,
                "f1_score": f1_sum_train / num_samples,
                "accuracy": acc_sum_train / num_samples,
            })

        epoch_loss = running_loss / num_samples
        epoch_f1 = f1_sum_train / num_samples
        epoch_acc = acc_sum_train / num_samples

        return epoch_loss, epoch_f1, epoch_acc

    def _validate(self, epoch):
        self.model.eval()
        val_running_loss = 0.0
        f1_sum_val = 0.0
        acc_sum_val = 0.0
        num_samples = 0

        validation_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                              desc=f'Validation Epoch {epoch + 1}/{self.num_epochs}')

        for step, (sentinel1, sentinel2, labels) in validation_bar:
            sentinel1 = sentinel1.to(self.device, dtype=torch.float32)  
            sentinel2 = sentinel2.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(sentinel1, sentinel2)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue  

                    loss = self.criterion(outputs, labels)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue  

                batch_size = sentinel1.size(0)
                val_running_loss += loss.item() * batch_size
                num_samples += batch_size

                f1_sum_val += self.metric_calculator.f1_score(outputs, labels).item() * batch_size
                acc_sum_val += self.metric_calculator.pixel_accuracy(outputs, labels).item() * batch_size

                validation_bar.set_postfix({
                    "val_loss": val_running_loss / num_samples,
                    "f1_score": f1_sum_val / num_samples,
                    "accuracy": acc_sum_val / num_samples,
                })

        avg_val_loss = val_running_loss / num_samples
        avg_val_f1 = f1_sum_val / num_samples
        avg_val_acc = acc_sum_val / num_samples

        return avg_val_loss, avg_val_f1, avg_val_acc

    def _check_early_stopping(self, val_loss, val_acc, val_f1, epoch):
        try:
            # Calculate weighted metric
            #weighted_metric = (self.loss_weight * val_loss) + \
            #                  (self.acc_weight * (1 - val_acc)) + \
            #                  (self.f1_weight * (1 - val_f1))
            
            save_checkpoint = False
            
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint = True

            #if self.best_val_f1 is None or val_f1 > self.best_val_f1:
            #    self.best_val_f1 = val_f1
            #    save_checkpoint = True

            #if self.best_val_acc is None or val_acc > self.best_val_acc:
            #    self.best_val_acc = val_acc
            #    save_checkpoint = True

            #if self.best_weighted_metric is None or weighted_metric < self.best_weighted_metric:
            #    self.best_weighted_metric = weighted_metric
            if save_checkpoint == True:
                self.patience_counter = 0

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_acc': val_acc
                }, self.checkpoint_path)
                print(f'Saving model checkpoint at epoch {epoch + 1}')

            else:
                self.patience_counter += 1

        except Exception as e:
            print(f'Error during early stopping check: {e}')
            raise