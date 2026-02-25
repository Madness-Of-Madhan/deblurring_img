import torch
from concurrent.futures import ThreadPoolExecutor

class InferenceEngine:
    def __init__(self, mprnet, deblurgan, unet):
        self.mprnet = mprnet
        self.deblurgan = deblurgan
        self.unet = unet

        self.pool = ThreadPoolExecutor(max_workers=3)

    def run_all(self, x):
        with torch.no_grad():
            f1 = self.pool.submit(self.mprnet, x)
            f2 = self.pool.submit(self.deblurgan, x)
            f3 = self.pool.submit(self.unet, x)

        return {
            "mprnet": f1.result(),
            "deblurgan_v2": f2.result(),
            "unet_patchgan": f3.result()
        }
