import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.autograd import Variable
from layers_torch import *
import time
import numpy
import os, sys
import random
import matplotlib.pyplot as plt
import argparse as Ap
import ipdb;bp=ipdb.set_trace


argp = Ap.ArgumentParser()
argp.add_argument("-s", "--seed",           default=0x6a09e667f3bcc908, type=long,
    help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
argp.add_argument("--model",                default="sparseattn",       type=str,
    choices=["sparseattn", "sparseattn_predict", "trunc", "baseline"],
    help="Model Selection.")
argp.add_argument("-n", "--num-epochs",     default=20,                 type=int,
    help="Number of epochs")
argp.add_argument("--bs",                   default=64,                 type=int,
    help="Training Batch Size")
argp.add_argument("--vbs",                  default=64,                 type=int,
    help="Validation Batch Size")
argp.add_argument("--rnn-dim",              default=128,                type=int,
    help="RNN hidden state size")
argp.add_argument("--rnn-layers",           default=2,                  type=int,
    help="Number of RNN layers")
argp.add_argument("--attk",                 default=2,                  type=int,
    help="Attend every K timesteps")
argp.add_argument("--topk",                 default=10,                 type=int,
    help="Attend only to the top K most important timesteps.")
argp.add_argument("--trunc",                default=10,                 type=int,
    help="Truncation length")
argp.add_argument("-T",                     default=200,                type=int,
    help="Copy Distance")
argp.add_argument("--clipnorm", "--cn",     default=1.0,              type=float,
    help="The norm of the gradient will be clipped at this magnitude.")
argp.add_argument("--lr",                   default=1e-3,               type=float,
    help="Learning rate")
argp.add_argument("--cuda",                 default=None,               type=int,
    nargs="?", const=0,
    help="CUDA device to use.")
argp.add_argument("--reload",               action="store_true",
    help="Whether to reload the network or not.")
argp.add_argument("--predict_m",              default=20,                type=int,
    help="predict m steps forward for hidden states")

d = argp.parse_args(sys.argv[1:])


numpy.random.normal(d.seed & 0xFFFFFFFF)
torch.manual_seed  (d.seed & 0xFFFFFFFF)
if d.cuda is not None:
    torch.cuda.manual_seed_all(d.seed & 0xFFFFFFFF)


input_size       = 1
rnn_dim          = d.rnn_dim
num_layers       = d.rnn_layers
num_classes      = 10
batch_size       = d.bs
valid_batch_size = d.vbs
num_epochs       = d.num_epochs
lr               = d.lr
n_words          = 2
maxlen           = 785
dictionary       = 'dict_bin_mnist.npz'
truncate_length  = d.trunc
T                = d.T
n_train          = 5000 * 128
n_test           = 512
n_sequence       = 10
attn_every_k     = d.attk
re_load          = d.reload
top_k            = d.topk
clip_norm        = d.clipnorm
predict_m        = d.predict_m
hist_valid_loss  = 1.0
hist_part_loss   = 5.0
hist_part_acc    = 1.0
beta             = 0.5

if not os.path.isdir("copying_logs"):
    os.mkdir("copying_logs")


def copying_data(T, n_data, n_sequence):
    seq = numpy.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = numpy.zeros((n_data, T-1))
    zeros2 = numpy.zeros((n_data, T))
    marker = 9 * numpy.ones((n_data, 1))
    zeros3 = numpy.zeros((n_data, n_sequence))

    x = numpy.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = numpy.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
    x = x.reshape(x.shape[0] / batch_size, batch_size, x.shape[1])
    y = y.reshape(y.shape[0] / batch_size, batch_size, y.shape[1])
    return x, y

train_x, train_y = copying_data(T, n_train, n_sequence)
test_x, test_y = copying_data(T, n_test, n_sequence)



if   d.model == "sparseattn":
    rnn = self_LSTM_sparse_attn(input_size, rnn_dim, num_layers, num_classes, truncate_length=truncate_length, top_k = top_k ,attn_every_k=attn_every_k) #, block_attn_grad_past=True)
elif d.model == "trunc":
    rnn = RNN_LSTM_truncated   (input_size, rnn_dim, num_layers, num_classes, truncate_length=truncate_length)
elif d.model == "baseline":
    rnn = RNN_LSTM             (input_size, rnn_dim, num_layers, num_classes)
elif d.model == "sparseattn_predict":
    rnn = self_LSTM_sparse_attn_predict (input_size, rnn_dim, num_layers, num_classes, truncate_length=truncate_length, top_k = top_k ,attn_every_k=attn_every_k, predict_m = predict_m)


if d.cuda is None:
    rnn.cuda()
    # rnn.cpu() # FIXME: Some day this should be uncommented!
else:
    rnn.cuda(d.cuda)


criterion = nn.CrossEntropyLoss()
l2_crit = nn.MSELoss()

opt = torch.optim.Adam(rnn.parameters(), lr=lr)

model_id, model_log, = rnn.print_log()

# model_log += ' clip norm ' + str(clip_norm)

model_id = 'T_' + str(T) + model_id + '_rnn_dim_'+ str(rnn_dim) + '_' + str(random.randint(1000, 9999))
folder_id = 'copying_logs/' + model_id

# if re_load is True
if re_load:
    model_id = 'copying_logs/best/T_200_LSTM-SAB_topk_5_truncate_length_1_norm-clip_0.5_9303'

os.mkdir(folder_id)
file_name = os.path.join(folder_id, model_id + '.txt')
model_file_name = os.path.join(folder_id, model_id + '.pkl')
attn_file =  os.path.join(folder_id, model_id + '.npz')

log_  = ''
log_ += 'Invocation:          '+' '.join(sys.argv)+'\n'
log_ += 'Timestamp:           '+time.asctime()+'\n'
log_ += 'SLURM Job Id:        '+str(os.environ.get('SLURM_JOBID', '-'))+'\n'
log_ += 'Start training ...T: ' +  str(T) + '...' + model_log +'....learning rate: ' + str(lr)

sys.stdout.write(log_+'\n')
sys.stdout.flush()
with open(file_name, 'a') as f:
    f.write(log_)


def save_param(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)

def load_param(model, model_file_name):
    model.load_state_dict(torch.load(model_file_name))

def attention_viz(attention_timestep, filename):
    # visualize attention
    max_len = attention_timestep[-1].cpu().data.numpy().shape[0]
    attn_all = []
    for attn in attention_timestep:
        attn = attn.cpu().data.numpy() 
        attn = numpy.append(attn, numpy.zeros(max_len - len(attn)))
        attn_all.append(attn)
    attn_all = numpy.asarray(attn_all)
    fig = plt.figure()
    cax = plt.matshow(attn_all, cmap=plt.cm.gray_r)
    plt.colorbar(cax)
    filename += '_attention.png'
    plt.savefig( os.path.join(folder_id, filename))
    plt.close()

def printgradnorm(self, grad_input, grad_output):
    '''print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size()) '''
    print('grad_input norm:', grad_input[0].data.norm())
    #print('grad_output norm:', grad_output[0].data.norm())


# rnn.fc.register_backward_hook(printgradnorm)

def evaluate_valid(valid_x, valid_y, hist_valid_loss, hist_part_loss, hist_part_acc):
    valid_loss = []
    valid_acc = []
    part_valid_loss = []
    i = 0
    for x in valid_x:
        y = valid_y[i]
        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = numpy.asarray(y, dtype=numpy.float32)
        y = torch.from_numpy(y)
        y = y.view(y.size()[0], y.size()[1], input_size)
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        if d.model == "sparseattn": 
            outputs, attn_w_viz = rnn(images)

        elif d.model == "sparseattn_predict":
            outputs, attn_w_viz, predicted_h, real_h = rnn(images)
        else:
            outputs = rnn(images)
        
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        #acc =  torch.equal(outputs.max(dim=2)[1][:,-10:,:] , labels[:,-10:,:])
        # acc = float(acc.data[0]) / (batch_size * 784 )
        acc = (outputs.max(dim=2)[1][:,-10:,:].data == labels[:,-10:,:].data).sum()
        acc = acc * 1.0 / (batch_size * 10)
        valid_acc.append(acc)
        part_shp = outputs.size()
        part_labels = labels[:, -10:, :].contiguous().view([part_shp[0] * 10])
        part_outputs = outputs[:,-10:, :].contiguous().view([part_shp[0] * 10, part_shp[2]])
        loss = criterion(outputs_reshp, labels_reshp)
        part_loss = criterion(part_outputs, part_labels)
        part_valid_loss.append(float(part_loss.data[0]))
        valid_loss.append(float(loss.data[0]))

        i += 1
    avg_valid_loss = numpy.asarray(valid_loss).mean()
    avg_part_loss = numpy.asarray(part_valid_loss).mean()
    print "last batch of valid outputs ", (outputs.max(dim=2)[1][:,-10:,:].data[0])
    print "last batch of targets ", labels[:,-10:,:].data[0]
    avg_part_acc = numpy.asarray(valid_acc).mean() * 100

    log_line = model_log + 'copyiing task, T: ' +  str(T) +'  rnn dim ' + str(rnn_dim) +  ' Epoch [%d/%d] Best valid loss: %.3f, average Loss: %.3f,  part loss: %.3f, average accuracy: %.3f, validation ' %(epoch, num_epochs, hist_valid_loss, avg_valid_loss , avg_part_loss, avg_part_acc) +'\n'

    print  (log_line)
    if avg_part_acc < hist_part_acc:
        save_param(rnn, model_file_name)
        hist_part_acc = avg_part_acc
        hist_part_loss = avg_part_loss
        hist_valid_loss = avg_valid_loss
    with open(file_name, 'a') as f:
        f.write(log_line)
    return (hist_valid_loss, hist_part_loss, hist_part_acc)        

if re_load and os.path.exists(model_file_name):
    load_param(rnn, model_file_name)
    print '--- Evaluating reloaded model ----'
    epoch = 0
    evaluate_valid(test_x, test_y, hist_part_loss, hist_part_acc)





def print_norm():
    param_norm = []
    for param in rnn.parameters():
        norm = param.grad.norm(2).data[0]/ numpy.sqrt(numpy.prod(param.size()))
        #print param.size()
        param_norm.append(norm)

    return param_norm


for epoch in range(num_epochs):
    i = 0
    for x in train_x:
        t = -time.time()
        y = train_y[i]
        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = numpy.asarray(y, dtype=numpy.float32)
        y = torch.from_numpy(y)
        y = y.view(y.size()[0], y.size()[1], input_size)
        
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        if  d.model == "sparseattn":
            outputs, attn_w_viz = rnn(images)
        elif d.model == "sparseattn_predict":
            outputs, attn_w_viz, predicted_h, real_h = rnn(images)
        else:
            outputs = rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1]) 
        loss = criterion(outputs_reshp, labels_reshp)
        if d.model == "sparseattn_predict":
            predict_loss = ((predicted_h[:, : -predict_m,:] - real_h[:,predict_m :,:].clone() ) ** 2).mean()
            loss += beta * predict_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(rnn.parameters(), clip_norm)
        opt.step()

        t += time.time()
        
        if  d.model == "sparseattn" and ((i+1) % 1000 == 0) :
            max_len = attn_w_viz[-1].cpu().data.numpy().shape[0] 
            attn_all = []
            for attn in attn_w_viz:
                attn = attn.cpu().data.numpy()
                attn = numpy.append(attn, numpy.zeros(max_len - len(attn)))
                attn_all.append(attn)
            with open(file_name, 'a') as f:
                for item in attn_all:
                    print ("%s\n" % item)
                    f.write("%s\n" % item)

            #import ipdb; ipdb.set_trace()
            #attn_all = numpy.asarray(attn_all).tostring()
            #log_line += attn_all + '\n'
                
        if (i+1) % 50 == 0:
            log_line = model_log + ' Epoch [%d/%d], Step %d  Loss: %f, batch_time: %f \n' %(epoch, num_epochs, i+1, loss.data[0], t)
            if (i + 1) % 500 == 0:
                print 'file saved at ', folder_id
            print (log_line)
            with open(file_name, 'a') as f:
                f.write(log_line)
                
            #print(print_norm())
            

        if  ((i + 1) % 200 == 0):
            hist_valid_loss, hist_part_loss, hist_part_acc = evaluate_valid(test_x, test_y, hist_valid_loss, hist_part_loss, hist_part_acc)
            if d.model == "sparseattn":
                attn_viz_file = model_id + '_epoch_'+str(epoch) + '_iter_' +str(i)
                attention_viz(attn_w_viz, attn_viz_file)

        i += 1
      
    # evaluate per epoch
    print '--- Epoch finished ----'
    hist_valid_loss, hist_part_loss, hist_part_acc = evaluate_valid(test_x, test_y, hist_valid_loss, hist_part_loss, hist_part_acc)
