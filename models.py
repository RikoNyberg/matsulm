import torch.nn as nn

####################################################################################
# MODEL
####################################################################################
# RNN based language model
class RNNLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            init_scale=None,
            init_bias=0,
            forget_bias=1,
        ):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True) #bidirectional=bidirectional)
        lstm_output_size = hidden_size #if not bidirectional else hidden_size * 2
        self.linear = nn.Linear(lstm_output_size, vocab_size)
        
        # Initializing weights/bias
        init_scale = 1.0/np.sqrt(hidden_size) if init_scale == None else init_scale
        for name, param in self.lstm.named_parameters(): # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
            if 'bias' in name:
                nn.init.constant_(param, init_bias)
            elif 'weight' in name:
                nn.init.uniform_(param, -init_scale, init_scale)
        
        # Setting Forget Gate bias
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
                
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Dropout vectors
        x = self.dropout(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)

        return out, (h, c)
