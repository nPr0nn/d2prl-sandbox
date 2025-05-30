import os
import csv
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

def JP_IO(img: np.ndarray) -> np.ndarray:
    """Simulate JPEG compression at a random quality using OpenCV only."""
    QF = random.choice([40, 50, 60, 70, 80, 90, 100])
    # encode to JPEG in-memory
    success, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    if not success:
        return img
    # decode back to array
    dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    # ensure same color order (BGR→RGB) if needed
    if dec.ndim == 3 and dec.shape[2] == 3:
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return dec

def noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise."""
    var = random.choice([0.009, 0.005, 0.0005])
    img_f = img.astype(np.float32) / 255.0
    n = np.random.normal(0, var**0.5, img_f.shape)
    out = np.clip(img_f + n, 0, 1.0)
    return (out * 255).astype(np.uint8)

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        h, w = self.size
        if img is not None:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img, mask = self.resize(img, mask)
        img = self.to_tensor(img).float()
        mask = self.to_tensor(mask).float()
        return img, mask

class Data(Dataset):
    def __init__(self, cfg, csv_path: str, folder_path: str, train_flag: bool = True):
        """
        cfg should have attribute `size` for transforms.
        csv_path must point to a CSV with columns 'file_path' and 'mask_path'.
        """
        self.flag = train_flag
        self.transform = CustomTransform(size=cfg.size)
        self.samples = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_p, m_p = os.path.join(folder_path, row['file_path']), os.path.join(folder_path, row['mask_path'])
                if os.path.isfile(img_p) and os.path.isfile(m_p):
                    self.samples.append((img_p, m_p))
                else:
                    raise FileNotFoundError(f"Missing file: {img_p} or {m_p}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = mask.astype(np.float32)
        if mask.ndim == 2:
            mask = np.stack([mask]*3, axis=2)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # augment
        if self.flag:
            mode = random.choice(['zero','zero','JC','JC','JC','NA'])
            if mode == 'JC':
                img = JP_IO(img)
            elif mode == 'NA':
                img = noise(img)

        # normalize mask
        if mask.max() > 1.0:
            mask = mask / 255.0

        # transform to tensors
        img_t, mask_t = self.transform(img, mask)
        name = os.path.basename(img_path)
        return img_t, mask_t, name

