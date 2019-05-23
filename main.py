# coding: utf-8
import argparse
import time
import math
import os
import pickle
import torch
import torch.nn as nn
import torch.onnx
import data
import models
import utils

def evaluate(args, model, corpus, data_source, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = utils.get_batch(args, data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(args, model, corpus, train_data, criterion, lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        # Break out earlyfor debugging
        if args.max_batches is not None and i >= args.max_batches:
            break

        data, targets = utils.get_batch(args, train_data, i)
        # Starting each batch, we detach the hidden state from how it was
        # previously produced. If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs
        # / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // args.bptt, lr,
                        elapsed * 1000 / args.log_interval, cur_loss,
                        math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def data_model(args):
    """
    Load data and model
    """
    train_data, val_data, test_data, corpus = data.get_data(args)

    print("Dataset: {}".format(args.dataset))
    print("Dataset path: {}".format(args.data))
    print("Dataset stats:")
    print("Train samples/tokens: {}".format(len(corpus.train)))
    print("Dev samples/tokens: {}".format(len(corpus.valid)))
    print("Test samples/tokens: {}".format(len(corpus.valid)))
    print("Vocabulary size: {}".format(len(corpus.dictionary.idx2word)))

    # Build or load the model
    ntokens = len(corpus.dictionary)
    model = models.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                        args.nlayers,args.dropout, args.tied).to(device)
    if args.load_model:
        model = torch.load(args.saved_model_path)
        print("Loaded saved model from: {}".format(args.saved_model_path))

    return train_data, val_data, test_data, corpus, model

def generate(args):
    # Get data and model
    train_data, val_data, test_data, corpus, model = data_model(args)

    # Predict random 5th sentence from test data
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, args.bptt):
            data, targets = utils.get_batch(args, test_data, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(test_data) - 1)
    print

def main(args):

    # Get data and model
    train_data, val_data, test_data, corpus, model = data_model(args)

    ###########################################################################
    # Train and evaluate
    ###########################################################################

    lr = args.lr
    best_val_loss = None

    criterion = nn.CrossEntropyLoss()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args, model, corpus, train_data, criterion, lr)
            val_loss = evaluate(args, model, corpus, val_data, criterion)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(\
                            epoch, (time.time() - epoch_start_time),
                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if args.save:
                    with open(args.save_path, 'wb') as f:
                        torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in
                # the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save_path, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(args, model, corpus, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        utils.export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='naive',
                        help='`naive`, `wiki-2`, `wiki-100`')
    parser.add_argument('--prepared_data', type=str, default='data/naive.pickle',
                        help='path of prepared data')
    parser.add_argument('--load_model', action='store_true',default=False,
                        help='load trained model')
    parser.add_argument('--saved_model_path', type=str,
                        default='saved/naive_model.pt',
                        help='path of prepared data')
    parser.add_argument('--test_percent', type=float, default=0.10,
                        help='Percent of dataset for testing, same used for valid')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='eval batch size')
    parser.add_argument('--max_batches', type=int, default=None, metavar='N',
                        help='max batches')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Whether to save the model while training')
    parser.add_argument('--generate', action='store_true', default=False,
                        help='Just generate with saved model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # CUDA
    args.cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Saving
    args.save_path = 'saved/' + args.dataset + '_model.pt'

    if args.generate:
        generate(args)
    else:
        main(args)
