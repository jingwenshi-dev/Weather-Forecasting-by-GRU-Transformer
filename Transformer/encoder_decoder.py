import torch.nn as nn
import torch
from components import PositionWiseFFN, AddNorm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, X, valid_lens, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, dec_state)


class VAE_EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, d_model):
        super(VAE_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_var = nn.Linear(d_model, d_model)
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)

        mu = self.fc_mu(enc_outputs)
        logvar = self.fc_var(enc_outputs)
        z = self.reparameterize(mu, logvar)
        dec_state = self.decoder.init_state(z, enc_valid_lens)
        return self.decoder(dec_X, dec_state), mu, logvar


class Latent(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(Latent, self).__init__()
        self.mean = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
        self.logvar = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
        self.mean_p = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
        self.logvar_p = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X, X_p):
        mu = self.mean(X)
        logvar = self.logvar(X)
        mu_p = self.mean_p(X_p)
        logvar_p = self.logvar_p(X_p)
        return mu, logvar, mu_p, logvar_p


class CVAE_EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, prior_encoder, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(CVAE_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.prior_encoder = prior_encoder
        self.latent = Latent(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / \
                      logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)

        mu = self.fc_mu(enc_outputs)
        logvar = self.fc_var(enc_outputs)
        z = self.reparameterize(mu, logvar)
        dec_state = self.decoder.init_state(z, enc_valid_lens)
        return self.decoder(dec_X, dec_state), mu, logvar


