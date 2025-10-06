import torch
import torchprofile
from datetime import datetime
from tqdm.auto import tqdm
from utils.utils import Metrics
from utils.log import Logger
import utils.config as config

class ModelTester:
    def __init__(self, model, test_loader, device, checkpoint):
        self.model = model
        self.test_loader = test_loader
        self.device = device

        self.logger = Logger('_')
        self.metric_calculator = Metrics(num_classes=config.NUM_CLASSES)

        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            raise

    def test(self, test_loader):
        self.model.eval()
        
        # Inisialisasi metrik global
        global_metrics = {
            'intersection': torch.zeros(self.metric_calculator.num_classes, device=self.device),
            'union': torch.zeros(self.metric_calculator.num_classes, device=self.device),
            'tp': torch.zeros(self.metric_calculator.num_classes, device=self.device),
            'fp': torch.zeros(self.metric_calculator.num_classes, device=self.device),
            'fn': torch.zeros(self.metric_calculator.num_classes, device=self.device),
            'correct_pixels': 0,
            'total_pixels': 0,
            'batch_count': 0
        }
        
        class_metrics = {
            f'class_{cls}': {'intersection': 0.0, 'union': 0.0, 'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
            for cls in range(self.metric_calculator.num_classes)
        }

        try:
            start_time = datetime.now()
            
            with torch.no_grad():
                test_bar = tqdm(test_loader, desc='Testing', leave=True)
                for sentinel1, sentinel2, labels in test_bar:
                    sentinel1 = sentinel1.to(self.device, dtype=torch.float16)
                    sentinel2 = sentinel2.to(self.device, dtype=torch.float16)
                    labels = labels.to(self.device, dtype=torch.long)
                    
                    # Forward pass
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(sentinel1, sentinel2)
                    else:
                        outputs = self.model(sentinel1, sentinel2)
                    
                    global_metrics['batch_count'] += 1
                    batch_size = sentinel1.size(0)
                    
                    # Validasi dimensi
                    assert outputs.shape[1] == self.metric_calculator.num_classes, \
                        f"Output channels {outputs.shape[1]} != num_classes {self.metric_calculator.num_classes}"
                    
                    # Hitung metrik per kelas
                    for cls in range(self.metric_calculator.num_classes):
                        pred_cls = (outputs[:, cls] > self.metric_calculator.threshold).float()
                        target_cls = (labels == cls).float() if labels.dim() == 3 else labels[:, cls].float()
                        
                        # Hitung komponen metrik
                        intersection = (pred_cls * target_cls).sum()
                        union = pred_cls.sum() + target_cls.sum() - intersection
                        tp = intersection
                        fp = pred_cls.sum() - intersection
                        fn = target_cls.sum() - intersection
                        
                        # Update metrik global
                        global_metrics['intersection'][cls] += intersection
                        global_metrics['union'][cls] += union
                        global_metrics['tp'][cls] += tp
                        global_metrics['fp'][cls] += fp
                        global_metrics['fn'][cls] += fn
                        
                        # Update metrik per kelas
                        class_metrics[f'class_{cls}']['intersection'] += intersection.item()
                        class_metrics[f'class_{cls}']['union'] += union.item()
                        class_metrics[f'class_{cls}']['tp'] += tp.item()
                        class_metrics[f'class_{cls}']['fp'] += fp.item()
                        class_metrics[f'class_{cls}']['fn'] += fn.item()
                    
                    # Hitung akurasi pixel
                    pred_labels = outputs.argmax(dim=1)
                    true_labels = labels.argmax(dim=1) if labels.dim() == 4 else labels
                    correct = (pred_labels == true_labels).sum().item()
                    total = true_labels.numel()
                    
                    global_metrics['correct_pixels'] += correct
                    global_metrics['total_pixels'] += total
                    
                    # Update progress bar
                    test_bar.set_postfix({
                        'batch': global_metrics['batch_count'],
                        'curr_acc': f"{correct/total:.4f}" if total > 0 else "0.0000"
                    })
            
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            
            # Hitung semua metrik akhir
            ## IoU calculations
            iou_per_class = self.metric_calculator.safe_divide(
                global_metrics['intersection'], 
                global_metrics['union']
            )
            
            ## mIoU versions
            miou_traditional = iou_per_class.mean().item()
            valid_classes = global_metrics['union'] > 0
            miou_valid = iou_per_class[valid_classes].mean().item() if valid_classes.any() else 0.0
            
            ## Weighted mIoU
            class_weights = global_metrics['union'] / global_metrics['union'].sum()
            miou_weighted = (iou_per_class * class_weights).sum().item()
            
            ## F1-score calculations
            precision = self.metric_calculator.safe_divide(
                global_metrics['tp'],
                global_metrics['tp'] + global_metrics['fp']
            )
            recall = self.metric_calculator.safe_divide(
                global_metrics['tp'],
                global_metrics['tp'] + global_metrics['fn']
            )
            f1_per_class = self.metric_calculator.safe_divide(
                2 * precision * recall,
                precision + recall
            )
            
            avg_f1 = f1_per_class.mean().item()
            weighted_f1 = (f1_per_class * class_weights).sum().item()
            
            ## Pixel accuracy
            pixel_acc = self.metric_calculator.safe_divide(
                global_metrics['correct_pixels'],
                global_metrics['total_pixels']
            )
            
            ## Throughput
            throughput = global_metrics['batch_count'] / elapsed_time
            
            # Print hasil utama
            print("\n=== Final Test Results ===")
            print(f"Traditional mIoU: {miou_traditional:.4f}")
            print(f"Valid Classes mIoU: {miou_valid:.4f}")
            print(f"Weighted mIoU: {miou_weighted:.4f}")
            print(f"Mean F1-score: {avg_f1:.4f}")
            print(f"Weighted F1-score: {weighted_f1:.4f}")
            print(f"Pixel Accuracy: {pixel_acc:.4f}")
            print(f"Throughput: {throughput:.2f} images/sec")
            print(f"Total time: {elapsed_time:.2f} seconds")
            
            # Print metrik per kelas
            print("\n=== Per-class Metrics ===")
            for cls in range(self.metric_calculator.num_classes):
                cls_iou = iou_per_class[cls].item()
                cls_f1 = f1_per_class[cls].item()
                cls_support = global_metrics['union'][cls].item()
                print(f"Class {cls}: IoU={cls_iou:.4f}, F1={cls_f1:.4f}, Support={cls_support:.0f}")
            
            # Hitung GFLOPS
            gflops = self._calculate_gflops((sentinel1, sentinel2))
            print(f"\nModel GFLOPS: {gflops:.4f}")
            
            # Log hasil
            self._log_test_results(
                0,  # test loss (jika ada)
                weighted_f1,
                pixel_acc,
                miou_weighted,  # Gunakan weighted sebagai metrik utama
                gflops,
                throughput
            )
            
            return {
                'miou_traditional': miou_traditional,
                'miou_valid': miou_valid,
                'miou_weighted': miou_weighted,
                'f1_mean': avg_f1,
                'f1_weighted': weighted_f1,
                'pixel_accuracy': pixel_acc,
                'throughput': throughput,
                'class_metrics': class_metrics
            }

        except Exception as e:
            print(f'Error during testing: {str(e)}')
            raise

    def _log_test_results(self, avg_test_loss, avg_test_f1, avg_test_acc, avg_test_miou, gflops, throughput):
        test_filename = 'test_log.csv'
        current_datetime = datetime.now()
        date = current_datetime.date()
        time = current_datetime.time()
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_channels = self.test_loader.dataset[0][0].shape[0]
        batch_size = self.test_loader.batch_size
        dataset = config.ROOT_DIR

        self.logger.log_test_results(
            test_filename, date, time, self.model.__class__.__name__, num_parameters, num_channels, batch_size, avg_test_loss, avg_test_f1, avg_test_acc, avg_test_miou, gflops, throughput, dataset)

    def _calculate_gflops(self, inputs):
        """
        Menghitung GFLOPs dari model dengan satu atau dua input (multi-encoder).
        
        Parameters:
            inputs (Tensor atau tuple/list of Tensors): Input yang akan digunakan untuk profiling.

        Returns:
            float: Nilai GFLOPs dari model.
        """
        import torchprofile

        # Ubah model dan input ke half precision untuk simulasi ringan
        self.model.half()

        # Jika input merupakan tuple/list, ubah seluruhnya ke half
        if isinstance(inputs, (tuple, list)):
            inputs = tuple(inp.half() for inp in inputs)
        else:
            inputs = inputs.half()

        # Hitung FLOPs menggunakan torchprofile
        try:
            flops = torchprofile.profile_macs(self.model, inputs)
        except Exception as e:
            self.model.float()
            raise RuntimeError(f"Gagal menghitung FLOPs: {e}")

        # Kembalikan model ke float precision
        self.model.float()

        # Konversi ke GFLOPs
        gflops = flops / 1e9
        return gflops
