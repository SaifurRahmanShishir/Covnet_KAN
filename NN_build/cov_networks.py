import torch
from kan import KAN


class CovNetShallowKAN(torch.nn.Module):
    def __init__(self, d, N, R, grid=3, k=3, seed=21):
        super().__init__()

        self.net = KAN(
            width=[d, R, N],
            grid=grid,
            k=k,
            seed=seed
        )

        self.params = list(self.parameters())

    def forward(self, u):
        device = next(self.parameters()).device
        u = u.to(device)
        return self.net(u).T

    def plot(self, *args, **kwargs):
        return self.net.plot(*args, **kwargs)






class CovNetDeepKAN(torch.nn.Module):
    def __init__(self, d, N, R, depth=3, hidden_dim=30, grid=3, k=3, seed=21):
        super().__init__()

        self.components = torch.nn.ModuleList()

        for r in range(R):
            width = [d] + [hidden_dim] * depth + [1]

            self.components.append(
                KAN(
                    width=width,
                    grid=grid,
                    k=k,
                    seed=seed + r
                )
            )

        self.final_layer = torch.nn.Linear(R, N, bias=False)
        self.params = list(self.parameters())

    def forward(self, u):
        device = next(self.parameters()).device
        u = u.to(device)

        component_outputs = []

        for net in self.components:
            g_r = net(u)
            component_outputs.append(g_r)

        G = torch.cat(component_outputs, dim=1)
        x_hat = self.final_layer(G)

        return x_hat.T

    def plot_component(self, r, *args, **kwargs):
        return self.components[r].plot(*args, **kwargs)






class CovNetDeepSharedKAN(torch.nn.Module):
    def __init__(self, d, N, R, depth=3, hidden_dim=30, grid=3, k=3, seed=21):
        super().__init__()

        width = [d] + [hidden_dim] * depth + [R]

        self.shared_net = KAN(
            width=width,
            grid=grid,
            k=k,
            seed=seed
        )

        self.final_layer = torch.nn.Linear(R, N, bias=False)
        self.params = list(self.parameters())

    def forward(self, u):
        device = next(self.parameters()).device
        u = u.to(device)

        h = self.shared_net(u)
        x_hat = self.final_layer(h)

        return x_hat.T

    def plot(self, *args, **kwargs):
        return self.shared_net.plot(*args, **kwargs)