import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.weighted_metric = 0

    def log_epoch(self, num_epochs, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc):
        train_loss = float(f"{train_loss:.4f}")
        val_loss = float(f"{val_loss:.4f}")
        train_f1 = float(f"{train_f1:.4f}")
        val_f1 = float(f"{val_f1:.4f}")
        train_acc = float(f"{train_acc:.4f}")
        val_acc = float(f"{val_acc:.4f}")
        self.weighted_metric = (0.2 * val_loss) + (0.3 * (1 - val_acc)) + (0.5 * (1 - val_f1))
        self.weighted_metric = float(f"{self.weighted_metric:.4f}")

        epoch_header = ["Num Epochs", "Weighted Metric", "Train Loss", "Val Loss", "Train F1", "Val F1", "Train PixAcc", "Val PixAcc"]

        epoch_data = [num_epochs, self.weighted_metric, train_loss, val_loss, train_f1, val_f1, train_acc, val_acc]

        write_header = not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(epoch_header)
            writer.writerow(epoch_data)

    def log_test_results(self, test_filename, date, time, model_type, num_parameters, num_channels, batch_size, test_loss, test_f1, test_acc, miou, gflops, throughput, dataset):
        test_loss = float(f"{test_loss:.5f}")
        test_f1 = float(f"{test_f1:.5f}")
        test_acc = float(f"{test_acc:.5f}")
        miou = float(f"{miou:.5f}")
        gflops = float(f"{gflops:.2f}")
        throughput = float(f"{throughput:.2f}")

        test_header = ["Date", "Time", "Model Type", "Num Parameters", "Num Channels", "Batch Size", "Test Loss", "F1 Score", "Pixel Acc", "mIoU", "gflops", "throughput", "dataset"]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date, time = current_time.split()
        test_data = [date, time, model_type, num_parameters, num_channels, batch_size, test_loss, test_f1, test_acc, miou, gflops, throughput, dataset]
        
        write_header = not os.path.exists(test_filename) or os.path.getsize(test_filename) == 0
        with open(test_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(test_header)
            writer.writerow(test_data)