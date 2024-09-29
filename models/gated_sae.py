from utils import *

class GatedSae(nn.Module):
    def __init__(self, img_size, expansion_factor):
        super(GatedSae, self).__init__()
        self.img_size = img_size
        self.act_size = torch.prod(torch.tensor(self.img_size)).item() # input_dim
        self.hidden_size = int(self.act_size*expansion_factor) # output_dim

        # unclear which weight initialization is chosen in the original implementation
        self.W_gate = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.hidden_size, self.act_size)))
        self.b_gate = nn.Parameter(torch.zeros(self.hidden_size))

        self.b_mag = nn.Parameter(torch.zeros(self.hidden_size))

        self.r_mag = nn.Parameter(torch.zeros(self.hidden_size)) # rescaling parameter
       
        self.decoder = nn.Linear(self.hidden_size, self.act_size)
        self.decoder.bias = nn.Parameter(torch.zeros(self.act_size))
        # dec_weight has shape (act_size, hidden_size) by same argument as above but all quantities reversed
        dec_weight = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.act_size, self.hidden_size)))
        # We initialize s.t. its columns (rows of the transpose) have unit norm
        # dim=0 --> across the rows --> normalize each column
        # If we consider the tranpose: dim=1 (dim=-1) --> across the columns --> normalize each row
        dec_weight.data[:] = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = dec_weight

    def forward(self, x):
        if len(x.shape) == 4:
            x_new = rearrange(x, 'b c h w -> (b h w) c')   
            transformed = True
        else:
            transformed = False
            x_new = x
        x_cent = x_new - self.decoder.bias

        pi_gate = F.linear(x_cent, self.W_gate, self.b_gate) # pre-activation of sub-layer f_gate
        # we don't use gradients of f_gate as we use it with the heaviside function
        f_gate = torch.heaviside(pi_gate, torch.tensor([0.5]).to(x.device)).detach()
        # second argument of heaviside is the value it should take at x=0, default is 0.5

        # weight sharing: we define W_mag in terms of W_gate
        W_mag = torch.exp(self.r_mag[:, None]) * self.W_gate
        f_mag = F.relu(F.linear(x_cent, W_mag, self.b_mag))

        encoder_output = f_gate * f_mag # elementwise multiplication

        decoder_output = self.decoder(encoder_output)

        # quantities that are used in the loss
        relu_pi_gate = F.relu(pi_gate)
        # pass relu_pi_gate through decoder but with frozen weights
        with torch.no_grad():
            via_gate = F.linear(relu_pi_gate, self.decoder.weight.clone().detach(), self.decoder.bias.clone().detach())
        
        return encoder_output, decoder_output, relu_pi_gate, via_gate