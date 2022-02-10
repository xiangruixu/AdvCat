from torch import nn
import torch
import torch.nn.functional as F
import random
from deepcase.decoders  import DecoderAttention, DecoderEvent
from deepcase.embedding import EmbeddingOneHot
from deepcase.encoders  import Encoder
from deepcase.loss      import LabelSmoothing
# from torchtrain import Module

class geneRNN(nn.Module):
    def __init__(self):
        super(geneRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.n_diagnosis_codes = 5
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, 30)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    # overload forward() method
    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)
        h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))
        x = self.dropout(x)
        logit = self.fc(x)
        logit = self.softmax(logit)
        return logit

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class IPSRNN(nn.Module):
    def __init__(self):
        super(IPSRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 70
        hidden_size = 70
        n_labels = 3
        self.n_diagnosis_codes = 1104
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, input_size)
        # self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)
        h0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        output, h_n = self.lstm(x)
        x, attn_weights = self.attention(output.transpose(0, 1))
        x = self.dropout(x)
        logit = self.fc(x)
        logit = self.softmax(logit)
        return logit

class DeepLog(nn.Module):
    def __init__(self, input_size=28, hidden_size=64, output_size=28, num_layers=2):
        """DeepLog model used for training and predicting logs.

            Parameters
            ----------
            input_size : int
                Dimension of input layer.

            hidden_size : int
                Dimension of hidden layer.

            output_size : int
                Dimension of output layer.

            num_layers : int, default=2
                Number of hidden layers, i.e. stacked LSTM modules.
            """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers

        # Initialise model layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True)
        self.out  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    ########################################################################
    #                       Forward through network                        #
    ########################################################################

    def forward(self, X):
        """Forward sample through DeepLog.

            Parameters
            ----------
            X : tensor
                Input to forward through DeepLog network.

            Returns
            -------
            result : tensor

            """

        X = X.permute(1,0, 2)
        hidden = self._get_initial_state(X)
        state  = self._get_initial_state(X)

        # Perform LSTM layer
        out, hidden = self.lstm(X, (hidden, state))
        out = self.out(out[:, -1, :])
        out = self.softmax(out)
        return out
    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def _get_initial_state(self, X):
        """Return a given hidden state for X."""
        # Return tensor of correct shape as device
        return torch.zeros(
            self.num_layers ,
            X.size(0)       ,
            self.hidden_size
        ).to(X.device)


class DeepCase(nn.Module):
    def __init__(self, input_size=28, output_size=28, hidden_size=128, num_layers=1,
                 max_length=10, bidirectional=False, LSTM=False):
        """DeepCase that learns to interpret context from security events.
            Based on an attention-based Encoder-Decoder architecture.

            Parameters
            ----------
            input_size : int
                Size of input vocabulary, i.e. possible distinct input items

            output_size : int
                Size of output vocabulary, i.e. possible distinct output items

            hidden_size : int, default=128
                Size of hidden layer in sequence to sequence prediction.
                This parameter determines the complexity of the model and its
                prediction power. However, high values will result in slower
                training and prediction times

            num_layers : int, default=1
                Number of recurrent layers to use

            max_length : int, default=2
                Maximum lenght of input sequence to expect

            bidirectional : boolean, default=False
                If True, use a bidirectional encoder and decoder

            LSTM : boolean, default=False
                If True, use an LSTM as a recurrent unit instead of GRU
            """

        # Initialise super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################

        # Create embedding
        self.embedding         = nn.Embedding(input_size, hidden_size)
        self.embedding_one_hot = EmbeddingOneHot(input_size)
        self.softmax = nn.Softmax(dim=-1)

        # Create encoder
        self.encoder = Encoder(
            embedding     = self.embedding_one_hot,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = bidirectional,
            LSTM          = LSTM
        )

        # Create attention decoder
        self.decoder_attention = DecoderAttention(
            embedding      = self.embedding,
            context_size   = hidden_size,
            attention_size = max_length,
            num_layers     = num_layers,
            dropout        = 0.1,
            bidirectional  = bidirectional,
            LSTM           = LSTM,
        )

        # Create event decoder
        self.decoder_event = DecoderEvent(
            input_size  = input_size,
            output_size = output_size,
            dropout     = 0.1,
        )

    ########################################################################
    #                        ContextBuilder Forward                        #
    ########################################################################

    def forward(self, X, y=None, steps=1, teach_ratio=0.5):
        """Forwards data through ContextBuilder.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Tensor of input events to forward.

            y : torch.Tensor of shape=(n_samples, steps), optional
                If given, use value of y as next input with probability
                teach_ratio.

            steps : int, default=1
                Number of steps to predict in the future.

            teach_ratio : float, default=0.5
                Ratio of sequences to hdfs_train that use given labels Y.
                The remaining part will be trained using the predicted values.

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, steps, output_size)
                The confidence level of each output event.

            attention : torch.Tensor of shape=(n_samples, steps, seq_len)
                Attention corrsponding to X given as (batch, out_seq, in_seq).
            """

        ####################################################################
        #                   Perform check on events in X                   #
        ####################################################################

        if X.max() >= self.embedding_one_hot.input_size:
            raise ValueError(
                "Expected {} different input events, but received input event "
                "'{}' not in expected range 0-{}. Please ensure that the "
                "ContextBuilder is configured with the correct input_size and "
                "output_size".format(
                self.embedding_one_hot.input_size,
                X.max(),
                self.embedding_one_hot.input_size-1,
            ))

        ####################################################################
        #                           Forward data                           #
        ####################################################################

        # Initialise save
        confidence = list()
        attention  = list()

        # Get initial inputs of decoder
        decoder_input  = torch.zeros(
            size       = (X.shape[1], 1),
            dtype      = torch.long,
            device     = X.device,
        )

        X_encoded, context_vector = self.encoder(X)

        # Loop over all targets
        for step in range(steps):
            # Compute attention
            attention_, context_vector = self.decoder_attention(
                context_vector = context_vector,
                previous_input = decoder_input,
            )

            # Compute event probability distribution
            confidence_ = self.decoder_event(
                X         = X_encoded,
                attention = attention_,
            )

            # Store confidence
            confidence.append(confidence_)
            # Store attention
            attention.append(attention_)
            logit = torch.stack(confidence, dim=1)
            logit = torch.squeeze(logit,dim=1)
            logit = self.softmax(logit)
            
            # Detatch from history
            if y is not None and random.random() <= teach_ratio:
                decoder_input = y[:, step]
            else:
                decoder_input = confidence_.argmax(dim=1).detach().unsqueeze(1)
        # Return result
        return logit

def model_file(Dataset, Model_Type):
    return Model[Dataset][Model_Type]


Splice_Model = {
    'Normal': './classifier/Adam_RNN.4832',
    'adversarial': './classifier/Adam_RNN.17490'
}

IPS_Model = {
    'Normal': './classifier/Mal_RNN.942',
    'adversarial': './classifier/Mal_adv.705',
}

HDFS_Model = {


}

Model = {
    'Splice': Splice_Model,
    'IPS': IPS_Model,
    'hdfs': HDFS_Model,
}

