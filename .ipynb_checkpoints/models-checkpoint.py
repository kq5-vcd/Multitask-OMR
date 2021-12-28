class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=0):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=dropout)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        
        del h
        del recurrent

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        
        del T
        del b

        return output
    

class CRNN(nn.Module):

    def __init__(self, input_size, img_height, nclass, conv_layers, rnn_hidden_states, dropout=0.5, kernel_size=3, pooling_size=2):
        super(CRNN, self).__init__()
        assert img_height % 16 == 0, 'img_height has to be a multiple of 16'

        self.cnn = nn.Sequential()
        for i in range(len(conv_layers)):
            input_channels = input_size if i == 0 else conv_layers[i-1]
            output_channels = conv_layers[i]

            self.cnn.add_module('conv{0}'.format(i),
                                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding = 1))
            self.cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(output_channels))
            self.cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            self.cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling_size, 2))
            
            del input_channels
            del output_channels
            
        self.rnn = nn.Sequential()
        for i in range(len(rnn_hidden_states)):
            inC = conv_layers[-1] if i == 0 else rnn_hidden_states[i-1]
            
            if i < len(rnn_hidden_states) - 1:
                self.rnn.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], 
                                                      rnn_hidden_states[i], dropout=dropout))
            else:
                self.rnn.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], nclass))
                
            del inC
        

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.view(b, c, -1)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        
        del b
        del c
        del h
        del w

        # rnn features
        output = self.rnn(conv)

        return output
    
class MTLCRNN(nn.Module):

    def __init__(self, input_size, img_height, nclass_sem, nclass_agn, conv_layers, rnn_hidden_states, dropout=0.5, kernel_size=3, pooling_size=2):
        super(MTLCRNN, self).__init__()
        assert img_height % 16 == 0, 'img_height has to be a multiple of 16'

        self.cnn = nn.Sequential()
        for i in range(len(conv_layers)):
            input_channels = input_size if i == 0 else conv_layers[i-1]
            output_channels = conv_layers[i]

            self.cnn.add_module('conv{0}'.format(i),
                                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding = 1))
            self.cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(output_channels))
            self.cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            self.cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling_size, 2))
            
            del input_channels
            del output_channels
            
        self.rnn_sem = nn.Sequential()
        for i in range(len(rnn_hidden_states)):
            inC = conv_layers[-1] if i == 0 else rnn_hidden_states[i-1]

            if i < len(rnn_hidden_states) - 1:
                self.rnn_sem.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], 
                                                      rnn_hidden_states[i], dropout=dropout))
            else:
                self.rnn_sem.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], nclass_sem))

            del inC

        self.rnn_agn = nn.Sequential()
        for i in range(len(rnn_hidden_states)):
            inC = conv_layers[-1] if i == 0 else rnn_hidden_states[i-1]

            if i < len(rnn_hidden_states) - 1:
                self.rnn_agn.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], 
                                                      rnn_hidden_states[i], dropout=dropout))
            else:
                self.rnn_agn.add_module('BiLSTM{0}'.format(i), 
                                    BidirectionalLSTM(inC, rnn_hidden_states[i], nclass_agn))

            del inC
        

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.view(b, c, -1)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size())
        
        del b
        del c
        del h
        del w

        # rnn features
        output1 = self.rnn_sem(conv)
        output2 = self.rnn_agn(conv)

        return output1, output2