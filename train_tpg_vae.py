import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"

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
from torch.nn import DataParallel
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.0001, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/tpg_vae', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='dataset/Cholec80', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='cholec80', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=20, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=16, help='dimensionality of z_t')
parser.add_argument('--l_dim', type=int, default=7, help='dimensionality of l_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='vgg', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=20, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
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

if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s'\
           % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        # opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)
        opt.log_dir = '%s/%s' % (opt.log_dir, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/ckpt/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+2*opt.z_dim+opt.l_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size//len(device_ids))
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size//len(device_ids), False)
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size//len(device_ids), False)
    posterior_motion = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size//len(device_ids), False)
    prior_motion = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size//len(device_ids), False)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model
    elif opt.image_width == 128:
        import models.dcgan_128 as model
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)

if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    encoder_motion = model.encoder_motion(opt.g_dim)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_motion_optimizer = opt.optimizer(prior_motion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_motion_optimizer = opt.optimizer(posterior_motion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_motion_optimizer = opt.optimizer(encoder_motion.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(
    #
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
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

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
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

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    # nsample = 20
    nsample = 1
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[2][i] for i in range(len(x[2]))]

    for s in range(nsample):
        # frame_predictor.module.hidden = frame_predictor.module.init_hidden()
        # posterior.module.hidden = posterior.module.init_hidden()
        # prior.module.hidden = prior.module.init_hidden()
        # posterior_motion.module.hidden = posterior_motion.module.init_hidden()
        # prior_motion.module.hidden = prior_motion.module.init_hidden()
        frame_predictor_hidden = frame_predictor.module.init_hidden(opt.batch_size, device)
        posterior_hidden = posterior.module.init_hidden(opt.batch_size, device)
        prior_hidden = prior.module.init_hidden(opt.batch_size, device)
        posterior_motion_hidden = posterior_motion.module.init_hidden(opt.batch_size, device)
        prior_motion_hidden = prior.module.init_hidden(opt.batch_size, device)
        gen_seq[s].append(x[2][0])
        x_in = x[2][0]
        x_in_motion = x[3][0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            h_motion = encoder_motion(x_in_motion)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
                h_motion, skip = h_motion
            else:
                h, _ = h
                h_motion = h_motion[0]
            # h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[2][i])
                # h_target = h_target[0].detach()
                h_target = h_target[0]
                h_motion_target = encoder_motion(x[3][i])[0]
                # z_t, _, _ = posterior(h_target)
                c_t, mu_c, logvar_c, posterior_hidden = posterior(h_target, posterior_hidden)
                m_t, mu_m, logvar_m, posterior_motion_hidden = posterior_motion(h_motion_target,
                                                                                posterior_motion_hidden)
                # prior(h)
                _, mu_p_c, logvar_p_c, prior_hidden = prior(h, prior_hidden)
                _, mu_p_m, logvar_p_m, prior_motion_hidden = prior_motion(h_motion, prior_motion_hidden)
                # frame_predictor(torch.cat([h, z_t], 1))
                l_t = x[1][i].to(device)
                h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, c_t, m_t, l_t], 1),
                                                                 frame_predictor_hidden)
                # x_in = x[i]
                x_in = x[2][i]
                x_in_motion = x[3][i]
                gen_seq[s].append(x_in)
            else:
                # z_t, _, _ = prior(h)
                p_c, mu_p_c, logvar_p_c, prior_hidden = prior(h, prior_hidden)
                p_m, mu_p_m, logvar_p_m, prior_motion_hidden = prior_motion(h_motion, prior_motion_hidden)
                # h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                l_t = x[1][i].to(device)
                h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, c_t, m_t, l_t], 1),
                                                                 frame_predictor_hidden)
                # x_in = decoder([h, skip]).detach()
                x_in = decoder([h_pred, skip])
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        # s_list = [min_idx,
        #           np.random.randint(nsample),
        #           np.random.randint(nsample),
        #           np.random.randint(nsample),
        #           np.random.randint(nsample)]
        s_list = [0]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    # frame_predictor.module.hidden = frame_predictor.module.init_hidden()
    # posterior.module.hidden = posterior.module.init_hidden()
    # posterior_motion.module.hidden = posterior_motion.module.init_hidden()
    frame_predictor_hidden = frame_predictor.module.init_hidden(opt.batch_size, device)
    posterior_hidden = posterior.module.init_hidden(opt.batch_size, device)
    posterior_motion_hidden = posterior_motion.module.init_hidden(opt.batch_size, device)
    gen_seq = []
    gen_seq.append(x[2][0])
    x_in = x[0]
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[2][i-1])
        h_target = encoder(x[2][i])
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        # h = h.detach()
        # h_target = h_target.detach()
        c_t, mu_c, logvar_c, posterior_hidden = posterior(h_target, posterior_hidden)

        h_motion = encoder_motion(x[3][i-1])
        h_motion_target = encoder_motion(x[3][i])[0]
        if opt.last_frame_skip or i < opt.n_past:
            h_motion, skip = h_motion
        else:
            h_motion = h_motion[0]
        m_t, mu_m, logvar_m, posterior_motion_hidden = posterior_motion(h_motion_target, posterior_motion_hidden)

        l_t = x[1][i].to(device)
        if i < opt.n_past:
            _, frame_predictor_hidden = frame_predictor(torch.cat([h, c_t, m_t, l_t], 1), frame_predictor_hidden)
            gen_seq.append(x[2][i])
        else:
            h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, c_t, m_t, l_t], 1), frame_predictor_hidden)
            x_pred = decoder([h_pred, skip])
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    posterior_motion.zero_grad()
    prior_motion.zero_grad()
    encoder.zero_grad()
    encoder_motion.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    # frame_predictor.module.hidden = frame_predictor.module.init_hidden()
    # posterior.module.hidden = posterior.module.init_hidden()
    # prior.module.hidden = prior.module.init_hidden()
    # posterior_motion.module.hidden = posterior_motion.module.init_hidden()
    # prior_motion.module.hidden = prior.module.init_hidden()
    # frame_predictor.module.init_hidden_new()
    # posterior.module.init_hidden_new()
    # prior.module.init_hidden_new()
    # posterior_motion.module.init_hidden_new()
    # prior.module.init_hidden_new()
    frame_predictor_hidden = frame_predictor.module.init_hidden(opt.batch_size, device)
    posterior_hidden = posterior.module.init_hidden(opt.batch_size, device)
    prior_hidden = prior.module.init_hidden(opt.batch_size, device)
    posterior_motion_hidden = posterior_motion.module.init_hidden(opt.batch_size, device)
    prior_motion_hidden = prior.module.init_hidden(opt.batch_size, device)

    mse = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[2][i-1])
        h_target = encoder(x[2][i])[0]
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h = h[0]
        c_t, mu_c, logvar_c, posterior_hidden = posterior(h_target, posterior_hidden)
        _, mu_p_c, logvar_p_c, prior_hidden = prior(h, prior_hidden)

        h_motion = encoder_motion(x[3][i-1])
        h_motion_target = encoder_motion(x[3][i])[0]
        if opt.last_frame_skip or i < opt.n_past:
            h_motion, skip = h_motion
        else:
            h_motion = h_motion[0]
        m_t, mu_m, logvar_m, posterior_motion_hidden = posterior_motion(h_motion_target, posterior_motion_hidden)
        _, mu_p_m, logvar_p_m, prior_motion_hidden = prior_motion(h_motion, prior_motion_hidden)

        l_t = x[1][i].to(device)
        h_pred, frame_predictor_hidden = frame_predictor(torch.cat([h, c_t, m_t, l_t], 1), frame_predictor_hidden)
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[2][i])
        kld += kl_criterion(mu_c, logvar_c, mu_p_c, logvar_p_c) + kl_criterion(mu_m, logvar_m, mu_p_m, logvar_p_m)

    mse = mse.sum()
    kld = kld.sum()
    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    posterior_motion_optimizer.step()
    prior_motion_optimizer.step()
    encoder_optimizer.step()
    encoder_motion_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    posterior_motion.train()
    prior_motion.train()
    encoder.train()
    encoder_motion.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    # progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        # progress.update(i+1)
        x = next(training_batch_generator)
        for i in range(4):
            x[i] = x[i].to(device)

        # train frame_predictor
        mse, kld = train(x)
        epoch_mse += mse
        epoch_kld += kld


    # progress.finish()
    # utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # plot some stuff
    frame_predictor.eval()
    #encoder.eval()
    #decoder.eval()
    posterior.eval()
    prior.eval()
    posterior_motion.eval()
    prior_motion.eval()

    x = next(testing_batch_generator)
    for i in range(4):
        x[i] = x[i].to(device)
    plot(x, epoch)
    plot_rec(x, epoch)

    if (epoch+1) % 5 == 0:
        print('log dir: %s' % opt.log_dir)
        # save the model
        torch.save({
            'encoder': encoder,
            'encoder_motion': encoder_motion,
            'decoder': decoder,
            'frame_predictor': frame_predictor,
            'posterior': posterior,
            'prior': prior,
            'posterior_motion': posterior_motion,
            'prior_motion': prior_motion,
            'opt': opt},
            '%s/ckpt/model_%d.pth' % (opt.log_dir, (epoch+1)))


