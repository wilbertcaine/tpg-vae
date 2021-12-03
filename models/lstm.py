import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        # self.hidden = self.init_hidden(device)
        # self.hidden_zeros = self.init_hidden(device)

    def init_hidden(self, batch_size, device):
        hidden = []
        for i in range(self.n_layers):
            # if torch.cuda.is_available():
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                           Variable(torch.zeros(batch_size, self.hidden_size).to(device))))
                # hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                #                Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
            # else:
            #     hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size)),
            #                    Variable(torch.zeros(self.batch_size, self.hidden_size))))
        return hidden

    def init_hidden_new(self):
        self.hidden = self.hidden_zeros

    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            # if iter == 1:
            #     self.hidden[i] = self.lstm[i](h_in, self.hidden_zeros[i])
            # else:
            #     self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            # self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            # h_in = self.hidden[i][0]
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        return self.output(h_in), hidden

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, reparam=True):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.reparam = reparam
        # self.hidden = self.init_hidden(device)
        # self.hidden_zeros = self.init_hidden(device)

    def init_hidden(self, batch_size, device):
        hidden = []
        for i in range(self.n_layers):
            # if torch.cuda.is_available():
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).to(device)),
                           Variable(torch.zeros(batch_size, self.hidden_size).to(device))))
                # hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                #                Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
            # else:
            #     hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size)),
            #                    Variable(torch.zeros(self.batch_size, self.hidden_size))))
        return hidden

    def init_hidden_new(self):
        self.hidden = self.hidden_zeros

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            # if iter == 1:
            #     self.hidden[i] = self.lstm[i](h_in, self.hidden_zeros[i])
            # else:
            #     self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            # self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            # h_in = self.hidden[i][0]
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        if self.reparam:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar, hidden
            
