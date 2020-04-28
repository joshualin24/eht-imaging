
# 2020-04-13 at Office
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import os, sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader



#import lenstronomy.Util.image_util as image_util
import tqdm
import pandas as pd
import numpy as np
import scipy as sp
import scipy.ndimage
import h5py
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import time
import gc
import datetime



imsize = 160

save_folder = '/media/joshua/Milano/uv_visibility/'

root_folder = "/media/joshua/HDD_fun2/1M_library_split/"
train_folder = "val_npy/"
test_folder = "test_npy/"
train_data = pd.read_csv(root_folder + train_folder + "val.csv")
test_data = pd.read_csv(root_folder + test_folder + "test.csv")
save_model_path = './saved_model/'

### hacking
def npy_save_txt(img, fname, xdim = 160, ydim = 160, source = 'M87', mjd=False, time=False):
    """Save image data to text file.
       Args:
            fname (str): path to output text file
            mjd (int): MJD of saved image
            time (float): UTC time of saved image
       Returns:
    """

    # Transform to Stokes parameters:

    RA = '12 h 30 m 49.4234 s'

    DEC = '12 deg 23 m 28.0437 s'

    # MJD: 0.000000

    RF: 227.0707

    # FOVX: 180 pix 0.000180 as

    # Coordinate values
    pdimas = 10**(-6)#psize/RADPERAS
    xs = np.array([[j for j in range(xdim)] for i in range(ydim)]).reshape(xdim*ydim,1)
    xs = pdimas * (xs[::-1] - xdim/2.0)
    ys = np.array([[i for j in range(xdim)] for i in range(ydim)]).reshape(xdim*ydim,1)
    ys = pdimas * (ys[::-1] - xdim/2.0)

    imvec = img[:, :, 0]
    qvec = img[:, :, 1]
    uvec = img[:, :, 2]
    vvec = img[:, :, 3]

    # If V values but no Q/U values, make Q/U zero
    if len(vvec) and not len(qvec):
        qvec = 0*vvec
        uvec = 0*vvec

    # Format Data
    if len(qvec) and len(vvec):
        outdata = np.hstack((xs, ys, (imvec).reshape(xdim*ydim, 1),
                                     (qvec).reshape(xdim*ydim, 1),
                                     (uvec).reshape(xdim*ydim, 1),
                                     (vvec).reshape(xdim*ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)  V (Jy/pixel)"

        fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f"

    elif len(qvec):
        outdata = np.hstack((xs, ys, (imvec).reshape(xdim*ydim, 1),
                                     (qvec).reshape(xdim*ydim, 1),
                                     (uvec).reshape(xdim*ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)"

        fmts = "%10.10f %10.10f %10.10f %10.10f %10.10f"

    else:
        outdata = np.hstack((xs, ys, (imvec).reshape(xdim*ydim, 1)))
        hf = "x (as)     y (as)       I (Jy/pixel)"
        fmts = "%10.10f %10.10f %10.10f"

    # Header
    if not mjd: mjd = float(mjd)
    if not time: time = time
    #mjd += (time/24.)
    #mjd = MJD
    head = ""
    head = ("SRC: %s \n" % source +
                "RA: " + RA + "\n" + "DEC: " + DEC + "\n" +
                "MJD: %.6f \n" % (float(mjd)) +
                "RF: 230.0000 GHz \n" +
                "FOVX: %i pix %f as \n" % (xdim, pdimas * xdim) +
                "FOVY: %i pix %f as \n" % (ydim, pdimas * ydim) +
                "------------------------------------\n" + hf)

    # Save
    np.savetxt(fname, outdata, header=head, fmt=fmts)
    return



#### dataset



def get_PSF(img, PSF_sigma =3.0):
    # 1.0 correspond to pix_res =0.02 arcsec
    PSF_image = gaussian_filter(img, PSF_sigma)
    return PSF_image


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
            transforms.ToTensor(), # scale to [0,1] and convert to tensor
            normalize,
            ])
target_transform = torch.Tensor
glo_batch_size = 1


class EHTDataset(Dataset): # torch.utils.data.Dataset
    def __init__(self, root_dir, train=True, imsize=160, FWHM = 0.0, transform=None, image_loader=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_folder = 'val_npy/'#'data_train'
        self.test_folder = 'test_npy/'#'data_test'
        self.FWHM = FWHM
        self.imsize = imsize


        if self.train:
            self.path = os.path.join(self.root_dir, self.train_folder)
            self.df = pd.read_csv(self.path + '/val.csv')
            #self.df = self.df.head(50000)
            #self.length = TRAINING_SAMPLES
        else:
            self.path = os.path.join(self.root_dir, self.test_folder)
            self.df = pd.read_csv(self.path + '/test.csv')
            self.df = self.df.head(100)
            #self.length = TESTING_SAMPLES

    def __getitem__(self, index):
        #images = list(self.loader(os.path.join(root_dir, 'Ma{}'.format(i)) for i in (['-0.94', '-0.5', '0', '+0.5', '+0.94'])))

        #print(self.df['ID'])
        fname = self.df['fname'].iloc[index]
        if "0.94" in fname[:7]:
            new_fname = fname[:7] + "/" + fname[7:]
        elif "0.5" in fname[:6]:
            new_fname = fname[:6] + "/" + fname[6:]
        else:
            new_fname = fname[:3] + "/" + fname[3:]
        flux = self.df['flux'].iloc[index]
        spin = self.df['spin'].iloc[index]
        mass = self.df['mass'].iloc[index]
        mdot = self.df['mdot'].iloc[index]
        inc = self.df['inc'].iloc[index]
        rlow = self.df['rlow'].iloc[index]
        rhigh = self.df['rhigh'].iloc[index]
        freq = self.df['freq'].iloc[index]
        ftot = self.df['ftot'].iloc[index]
        ftotunpol = self.df['ftotunpol'].iloc[index]
        time = self.df['time'].iloc[index]
        PA = self.df['PA'].iloc[index]
        sin_PA = np.sin(PA * np.pi/180.)
        cos_PA = np.cos(PA * np.pi/180.)
        #print("sin, cos", sin_PA, cos_PA)
        #print(fname)
        #print("ID:", ID.values[0])
        #print("flux:", flux)
        if flux == "sane":
            flux_value = 1.
        else:
            flux_value = 0.
        #print("flux_value:", flux_value)


        ###for testing
        mdot = np.log10(mdot)
        mass = np.log10(mass)

        #f = h5py.File(self.path + new_fname, 'r')
        img_path = self.path + "pol" + str(fname) + '.npy'
        pol_img = np.load(img_path)

        # ### Point Spread Function
        # PSF_image = get_PSF(img, PSF_sigma = self.FWHM/2.355)
        # img = PSF_image
        #
        #
        # ### normalizing
        #
        # img /= np.amax(img)
        #
        # ### Cropping image
        #
        # valid_a = [50, 55, 60]
        # a = np.random.choice(valid_a)
        # crop_img = PSF_image[80-a: 80+ a, 80- a: 80 + a]
        # ratio = a/80

        ###
        # sigma_to_noise_ratio = 100
        # total_flux = sum(sum(PSF_image))
        # N_pix = img.size
        # sigma_n = 0.0 #total_flux / (np.sqrt(N_pix) * sigma_to_noise_ratio)
        # gaussian_noise = sigma_n * np.random.randn(img.shape[0], img.shape[1])
        # Noisy_img = gaussian_noise + PSF_image
        # plt.imshow(Noisy_img)
        # plt.show()

        #img = scipy.ndimage.zoom(PSF_image, 1.4, order=1)
        #img = scipy.ndimage.zoom(img, 1.4, order=1)
        #img = scipy.ndimage.zoom(PSF_image, imsize/160, order=1)

#         plt.imshow(img)
#         plt.colorbar()
#         plt.show()
        npy_save_txt(pol_img, save_folder + str(fname) + '.txt', mjd = '48277.0000')
        # image = np.zeros((1, imsize, imsize))
        # for i in range(1):
        #     image[i, :, :] += img
        #return image, new_fname, flux_value, spin, mass, mdot, inc, rlow, rhigh, freq, ftot, ftotunpol, time, sin_PA, cos_PA
        return pol_img, spin

    def __len__(self):
        return self.df.shape[0]
        #return self.length


val_loader = torch.utils.data.DataLoader(EHTDataset(root_folder, train=True, transform=None, target_transform=None),
                    batch_size = glo_batch_size, shuffle = True
                    )

if __name__ == '__main__':
    print(val_loader)
    # for i in val_loader:
    #     print(i)
    #for batch_idx, (data, spin) in enumerate(tqdm(val_loader, total = len(val_loader))):
    for batch_idx, (data, spin) in enumerate(val_loader):
        if batch_idx % 100 == 0:
            print("batch_idx")
        if batch_idx > 1000:
            break
        gc.collect()
