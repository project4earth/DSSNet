# cleaner.py
import gc
import torch

class MemoryManager:
    def synchronize(self):
        torch.cuda.synchronize()

    def empty_cache(self):
        torch.cuda.empty_cache()

    def collect_garbage(self):
        gc.collect()

    def manage_memory(self):
        self.synchronize()
        self.empty_cache()
        self.collect_garbage()
