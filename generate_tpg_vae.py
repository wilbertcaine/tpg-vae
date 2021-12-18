import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
# import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from torch.nn import DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='dataset/Cholec80', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='logs/tpg_vae', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


opt = parser.parse_args()

cuda_is_available = torch.cuda.is_available()
if cuda_is_available:
    torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

str_ids = opt.gpu_ids.split(',')
device_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        device_ids.append(id)
device = torch.device('cuda:{}'.format(device_ids[0]) if cuda_is_available else "cpu")

os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor



# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path, map_location=device)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
# frame_predictor.cuda()
# posterior.cuda()
# prior.cuda()
# encoder.cuda()
# decoder.cuda()
frame_predictor = DataParallel(frame_predictor, device_ids = device_ids)
frame_predictor.to(device)
posterior = DataParallel(posterior, device_ids = device_ids)
posterior.to(device)
prior = DataParallel(prior, device_ids = device_ids)
prior.to(device)
posterior_motion = DataParallel(posterior_motion, device_ids = device_ids)
posterior_motion.to(device)
prior_motion = DataParallel(prior_motion, device_ids = device_ids)
prior_motion.to(device)
encoder = DataParallel(encoder, device_ids = device_ids)
encoder.to(device)
encoder_motion = DataParallel(encoder_motion, device_ids = device_ids)
encoder_motion.to(device)
decoder = DataParallel(decoder, device_ids = device_ids)
decoder.to(device)
mse_criterion = DataParallel(mse_criterion, device_ids = device_ids)
mse_criterion.to(device)

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------

def make_gifs(x, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    # progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(nsample):
        # progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    # progress.finish()
    # utils.clear_progressbar()

    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

for i in range(0, opt.N, opt.batch_size):
    # plot train
    train_x = next(training_batch_generator)
    make_gifs(train_x, i, 'train')

    # plot test
    test_x = next(testing_batch_generator)
    make_gifs(test_x, i, 'test')
    print(i)

