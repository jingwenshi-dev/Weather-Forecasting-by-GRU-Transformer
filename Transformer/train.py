import torch
import copy
from util import Accumulator, grad_clipping
torch.manual_seed(42)


def train_transformer(model, num_epochs, data_iter, loss, device,
                      corpus_size, target_vocab, mission, lr, code_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([target_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = model(X, dec_input, X_valid_len)
            print(Y_hat.shape)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch+1, num_epochs), end=' ')
            print("Loss: {:.4f}".format((metric[0] / metric[1])))
        if (metric[0] / metric[1]) < best_loss:
            best_loss = (metric[0] / metric[1])
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, code_type+'_transformer_'+mission+'_weights_'+str(corpus_size)+'_sentences_' +
                       str(num_epochs)+'_epochs_lr_'+str(lr)+'_on_'+str(torch.cuda.get_device_name(device))+'.pth')


def train_transformer_with_scheduler(model, num_epochs, data_iter, loss, device,
                                     corpus_size, target_vocab, mission, lr, mile_stone, gamma, code_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stone, gamma=gamma)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([target_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch+1, num_epochs), end=' ')
            print("Loss: {:.4f}".format((metric[0] / metric[1])))
        if (metric[0] / metric[1]) < best_loss:
            best_loss = (metric[0] / metric[1])
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, code_type+'_transformerSteppingLr_'+mission+'_weights_'+str(corpus_size)+'_sentences_' +
                       str(num_epochs)+'_epochs_initlr_'+str(lr)+'_on_'+str(torch.cuda.get_device_name(device))+'.pth')
        scheduler.step()


def train_vae_transformer_with_scheduler(model, num_epochs, data_iter, loss, device, corpus_size, target_vocab, mission,
                                         lr, mile_stone, gamma, code_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stone, gamma=gamma)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([target_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            dec_out, mu, log_var = model(X, dec_input, X_valid_len)
            l = loss(dec_out[0], Y, Y_valid_len, mu, log_var)
            l.sum().backward()
            grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}/{}.............'.format(epoch+1, num_epochs), end=' ')
            print("Loss: {:.4f}".format((metric[0] / metric[1])))
        if (metric[0] / metric[1]) < best_loss:
            best_loss = (metric[0] / metric[1])
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, code_type+'_vae_transformerSteppingLr_'+mission+'_weights_'+str(corpus_size)+
                       '_sentences_' + str(num_epochs)+'_epochs_initlr_'+str(lr)+'_on_'+
                       str(torch.cuda.get_device_name(device))+'.pth')
        scheduler.step()
