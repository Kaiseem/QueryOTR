import torch

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples = next(self.loader)
        except StopIteration:
            self.next_samples = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_samples.items():
                self.next_samples[k]=v.to(self.device)

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            if samples is not None:
                for k, v in samples.items():
                    v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples = next(self.loader)
                for k, v in self.next_samples.items():
                    self.next_samples[k] = v.to(self.device)
            except StopIteration:
                samples = None
        return samples


