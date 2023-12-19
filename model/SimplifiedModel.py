import torch
import torch.nn as nn

class PredictNoise(nn.Module):
    def __init__(self, input_channel=3, hidden_dim = 256, 
                 embedding_dim = 256, time_dimension = 128,
                 kernel_size = 3, device = 'cpu',
                 isConditional = False, class_size = None):
        super().__init__()

        self.input_channel = input_channel
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.device = device

        self.time_dimension = time_dimension
        self.time_embedding = nn.Linear(self.time_dimension, self.hidden_dim)

        # Conditional Model takes in the class and converts it as an embedding that gets appended 
        # to your image
        self.isConditional = isConditional
        if self.isConditional: self.class_embedding = nn.Embedding(class_size, embedding_dim)

        # Model Layer Definition happens here.
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channel, hidden_dim, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
        )

        self.block_5 = nn.Sequential(
            nn.ConvTranspose2d(2*hidden_dim, hidden_dim, kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channel, kernel_size),
            nn.LeakyReLU(),
        )

        self.last_conv = nn.Conv2d(2*input_channel, input_channel, kernel_size,  padding = "same")

    def encode_discrete(self, x, h, channels, encodeTime = True):
        """
        This is a abstract function to encode discrete feature i.e. time and label 
        h : [Batch-Size x 1]
        channels : is the number after the transformation of x i.e. its different from the channel x
        """
        h = self.time_embedding(h) if encodeTime else self.class_embedding(h.long())
        h = h.unsqueeze(2).unsqueeze(3) # [Batch-Size x 1 x 1 x1]
        h = h.expand(x.size(0), channels, x.size(2), x.size(3))
        return h

    def pos_encoding(self, time):
        """
        2D-Positional Embedding as defined in the classical attention model:
        https://arxiv.org/pdf/1706.03762.pdf
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.time_dimension, 2, device=self.device).float() / self.time_dimension)
        )
        pos_enc_a = torch.sin(time.repeat(1, self.time_dimension // 2) * inv_freq)
        pos_enc_b = torch.cos(time.repeat(1, self.time_dimension // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1) # [Batch-Size x self.time_dimension]
        return pos_enc

    def forward(self, x, time, label=None):
        time = time.unsqueeze(1) # [Batch-Size x 1]
        time = self.pos_encoding(time)# [Batch-Size x self.time_dimension]

        # 1st block is the transformation of the image without any positional/class embedding
        x1 = self.block_1(x)

        x2 = x1 + self.encode_discrete(x1, time, self.hidden_dim)
        if self.isConditional : x2 = x2 + self.encode_discrete(x1, label, self.hidden_dim, encodeTime=False)
        x2 = self.block_2(x2)

        x3 = x2 + self.encode_discrete(x2, time, self.hidden_dim)
        if self.isConditional : x3 = x3 +  self.encode_discrete(x2, label, self.hidden_dim, encodeTime = False)

        x3 = self.block_3(x3)

        # Convolution to higher up dimensions
        x4 = self.block_4(torch.cat((x3, x2), axis = 1))
        x5 = self.block_5(torch.cat((x4, x1), axis = 1))
        out = self.last_conv(torch.cat((x5, x), axis = 1))
        return out