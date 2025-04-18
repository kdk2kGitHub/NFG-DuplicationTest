from nfg.nfg_torch import *

class CondODENet(nn.Module):
    """
      Code extracted from https://github.com/djdanks/DeSurv
    """
    def __init__(self, cov_dim, layers, output_dim,
                 act = "ReLU", n = 15):
        super().__init__()
        self.output_dim = output_dim

        self.f = nn.Sequential(*create_representation(cov_dim + 1, layers + [output_dim], act, norm = False, last = nn.Softplus()))
        self.n = n

        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n, dtype = torch.float32)[None, :], requires_grad = False)
        self.w_n = nn.Parameter(torch.tensor(w_n, dtype = torch.float32)[None, :], requires_grad = False)

    def forward(self, x, horizon):
        u_n = self.u_n.to(x.device)
        w_n = self.w_n.to(x.device)
    
        tau = torch.matmul(horizon.unsqueeze(-1) / 2., 1 + u_n) # N x n (+ 1 to push integral in 0 2 and /2 to push in 0 - t)

        tau_ = torch.flatten(tau).unsqueeze(-1) # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        reppedx = torch.repeat_interleave(x, self.n, dim = 0)
        taux = torch.cat((tau_, reppedx), 1) # Nn x (d+1)

        f_n = self.f(taux).reshape((len(x), self.n, self.output_dim)) # N x n x d_out
        pred = horizon.unsqueeze(-1) / 2. * ((w_n[:, :, None] * f_n).sum(dim = 1))

        return torch.tanh(pred)


class DeSurvTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU', layers_surv = [100],
               risks = 1, optimizer = "Adam", n = 15, embedding = False, multihead = True):
    # No embedding in original paper.
    super().__init__()
    self.input_dim = inputdim
    self.risks = risks  # Competing risks
    self.optimizer = optimizer

    self.embedding = nn.Sequential(*create_representation(inputdim, layers + [inputdim], act)) if embedding else nn.Identity(inputdim)
    self.balance = nn.Sequential(*create_representation(inputdim, layers + [risks], act, last = nn.Softmax(dim = 1))) # Balance between risks

    self.odenet = nn.ModuleList(
                      [CondODENet(inputdim, layers_surv, 1, act, n = n) # Multihead (one for each outcome)
                  for _ in range(risks)]) if multihead \
                  else CondODENet(inputdim, layers_surv, risks, act, n = n)
    
    self.forward = self.forward_multihead if multihead else self.forward_single
    self.gradient = self.gradient_multihead if multihead else self.gradient_single

  def forward_single(self, x, horizon):
    x = self.embedding(x)
    balance = self.balance(x)
    Fr = self.odenet(x, horizon)
  
    return balance * Fr, balance, Fr, x
  
  def gradient_single(self, x_emb, horizon, k):
    return self.odenet.f(torch.cat((horizon.unsqueeze(1), x_emb), 1))[:, k]
  
  def forward_multihead(self, x, horizon):
    x = self.embedding(x)
    balance = self.balance(x)
    Fr = torch.cat([ode(x, horizon) for ode in self.odenet], 1)

    return balance * Fr, balance, Fr, x
  
  def gradient_multihead(self, x_emb, horizon, k):
    return self.odenet[k].f(torch.cat((horizon.unsqueeze(1), x_emb), 1))