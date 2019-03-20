import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt



def normal(tensor, mean=0, std=1):
    """Fills the input Tensor or Variable with values drawn from a normal distribution with the given mean and std
    Args:
        tensor: a n-dimension torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor
    else:
        return tensor.normal_(mean, std)


def uniform(tensor, a=0, b=1):
    """Fills the input Tensor or Variable with values drawn from the uniform
    distribution :math:`U(a, b)`.
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor

    return tensor.uniform_(a, b)


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out

    def print_log(self):
        model_name = '_regular-LSTM_'
        model_log = ' Regular LSTM.......'
        return (model_name, model_log)  


class RNN_LSTM_truncated (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=1):
        super(RNN_LSTM_truncated, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            if (i  + 1) % self.truncate_length == 0 :
                h_t , c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
                #c_t = c_t.detach()
            else:
                h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out
       
    def print_log(self):
        model_name = '_trun-LSTM_trun_len_'+str(self.truncate_length)
        model_log = ' trun LSTM.....trun_len:'+str(self.truncate_length)
        return (model_name, model_log)



class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            #delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements 
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
        
        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)
        return attn_w_normalize



def attention_visualize(attention_timestep, filename):
    # visualize attention
    plt.matshow(attention_timestep)
    filename += '_attention.png'
    plt.savefig(filename)
    
class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, predict_m = 10, block_attn_grad_past=False, attn_every_k=1, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.predict_m            = nn.Linear(hidden_size, hidden_size)


    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k)# + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        predict_h   = Variable(torch.zeros(batch_size, hidden_size).cuda())

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)

        outputs     = []
        attn_all    = []
        attn_w_viz  = []
        predicted_all = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            input_t    = input_t.contiguous().view(batch_size, input_size)
            h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            predict_h  = self.predict_m(h_t.detach())
            predicted_all.append(predict_h)


            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w     = torch.mm(mlp_h_attn, self.w_t)
                attn_w     = attn_w.view(batch_size, remember_size, 1)

            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            attn_w  = self.sparse_attn(attn_w)
            attn_w  = attn_w.view(batch_size, remember_size, 1)

            if i >= 100:
                attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        predicted_all = torch.stack(predicted_all, 1)
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        h_outs    = outputs.detach()
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz, predicted_all, h_outs)


class self_LSTM_sparse_attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, truncate_length=100, block_attn_grad_past=False, print_attention_step = 1, attn_every_k=1, top_k = 5):
        # latest sparse attentive backprop implementation
        super(self_LSTM_sparse_attn, self).__init__()
        self.hidden_size          = hidden_size
        self.num_layers           = num_layers
        self.num_classes          = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length      = truncate_length
        self.lstm1                = nn.LSTMCell(input_size, hidden_size)
        self.fc                   = nn.Linear(hidden_size * 2, num_classes)
        self.attn_every_k         = attn_every_k
        self.top_k                = top_k
        self.tanh                 = torch.nn.Tanh()
        self.w_t                  = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean = 0.0, std = 0.01)) #nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        self.sparse_attn          = Sparse_attention(top_k = self.top_k)
        self.atten_print          = print_attention_step

    def print_log(self):
        model_name = '_LSTM-sparse_attn_topk_attn_in_h' + str(self.top_k) + '_truncate_length_'+str(self.truncate_length) +'attn_everyk_' + str(self.attn_every_k)# + '_block_MLP_gradient_' + str(self.block_attn_grad_past)
        model_log = ' LSTM Sparse attention in h........topk:' + str(self.top_k)  +'....attn_everyk_' + str(self.attn_every_k) + '.....truncate_length:'+str(self.truncate_length)
        return (model_name, model_log)

    def forward(self, x):
        batch_size  = x.size(0)
        time_size   = x.size(1)
        input_size  = x.size(2)
        hidden_size = self.hidden_size
        h_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())
        c_t         = Variable(torch.zeros(batch_size, hidden_size).cuda())

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old       = h_t.view(batch_size, 1, hidden_size)
        
        outputs     = []
        attn_all    = []
        attn_w_viz  = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            remember_size = h_old.size(1)
            
            input_t    = input_t.contiguous().view(batch_size, input_size)
            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = self.lstm1(input_t, (h_t.detach(), c_t.detach()))
            
            else:
                # Feed LSTM Cell
                h_t, c_t   = self.lstm1(input_t, (h_t, c_t))
            
            # Broadcast and concatenate current hidden state against old states
            h_repeated =  h_t.unsqueeze(1).repeat(1, remember_size, 1)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)
            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()
            
            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)
            if False: # PyTorch 0.2.0
                attn_w     = torch.matmul(mlp_h_attn, self.w_t)
            else:     # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size*remember_size, 2*hidden_size)
                attn_w     = torch.mm(mlp_h_attn, self.w_t)
                attn_w     = attn_w.view(batch_size, remember_size, 1)
            
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w  = attn_w.view(batch_size, remember_size)
            attn_w  = self.sparse_attn(attn_w)
            attn_w  = attn_w.view(batch_size, remember_size, 1)
            
            if self.atten_print >= (time_size - i - 1) :
                attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w  = attn_w.repeat(1, 1, hidden_size)
            h_old_w = attn_w * h_old
            attn_c  = torch.sum(h_old_w, 1).squeeze(1)
            
            # Feed attn_c to hidden state h_t
            h_t    += attn_c

            #
            # At regular intervals, remember a hidden state.
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)
            
            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        outputs   = torch.stack(outputs,  1)
        attn_all  = torch.stack(attn_all, 1)
        outputs   = torch.cat  ((outputs, attn_all), 2)
        shp = outputs.size()
        out = outputs.contiguous().view(shp[0] *shp[1] , shp[2])
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return (out, attn_w_viz)


