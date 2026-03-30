# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


# Provided code
class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.
        Parameters:
        M: [int] Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """Variational Autoencoder with a single decoder."""

    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """Train a VAE model."""
    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"epoch={epoch}, step={step}, loss={loss:.1f}"
                    )
                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(f"Stopping at epoch {epoch}, loss={loss:.1f}")
                break


# Part A

def compute_curve_energy(curve, decoder):
    # page 73-76 in the book
    means = decoder.decoder_net(curve)

    means_flat = means.view(means.shape[0], -1)

    diffs = means_flat[1:] - means_flat[:-1]

    energy = (diffs ** 2).sum()
    return energy


def compute_geodesic(z_start, z_end, energy_fn, num_t=20, num_steps=300, lr=0.01):
    #page 74 in the book
    device = z_start.device

    t_vals = torch.linspace(0, 1, num_t, device=device)
    with torch.no_grad():
        curve_init = torch.stack([(1 - t) * z_start + t * z_end for t in t_vals]) # Piecewise linear

    inner = curve_init[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([inner], lr=lr) # ADAM or LBFGS

    for _ in range(num_steps):
        optimizer.zero_grad()

        curve = torch.cat([
            z_start.unsqueeze(0),   
            inner,                  
            z_end.unsqueeze(0)     
        ], dim=0)

        energy = energy_fn(curve)
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        curve = torch.cat([
            z_start.unsqueeze(0),
            inner.detach(),
            z_end.unsqueeze(0)
        ], dim=0)

    return curve

# part B

class EnsembleVAE(nn.Module):
    def __init__(self, prior, encoder, decoders):
        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)

    def elbo(self, x, decoder_idx):
        q = self.encoder(x)
        z = q.rsample()
        decoder = self.decoders[decoder_idx]
        return torch.mean(
            decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )

    def forward(self, x):
        dec = len(self.decoders)
        decoder_idx = torch.randint(0,dec, (1,)).item()
        return -self.elbo(x, decoder_idx)


def train_ensemble_vae(prior, encoder_fn, decoder_fn, decoderamount, data_loader, epochs, D, device, lrate = 1e-3, folder = None):
    num_steps = len(data_loader) * epochs * decoderamount
    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)
    for m in range(D):
        epoch = 0
        model = EnsembleVAE(
                prior=prior,
                encoder=GaussianEncoder(encoder_fn()),
                decoders=[GaussianDecoder(decoder_fn()) for _ in range(decoderamount)]
                ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                try:
                    x = next(iter(data_loader))[0]
                    x = noise(x.to(device))
                    optimizer.zero_grad()
                    loss = model(x)
                    loss.backward()
                    optimizer.step()

                    if step % 5 == 0:
                        loss = loss.detach().cpu()
                        pbar.set_description(
                            f"epoch={epoch}, step={step}, loss={loss:.1f}"
                        )
                    if (step + 1) % len(data_loader) == 0:
                        epoch += 1
                except KeyboardInterrupt:
                    print(f"Stopping at epoch {epoch}, loss={loss:.1f}")
                    break
        torch.save(model.state_dict(), f"{folder}/model_{decoderamount}dec_{m}.pt")


def compute_ensemble_energy(curve, decoders, num_mc_samples=10):
    T = curve.shape[0]
    num_decoders = len(decoders)
    energy = torch.tensor(0.0, device=curve.device)
    for i in range(T - 1):
        step_energy = torch.tensor(0.0, device=curve.device)
        for _ in range(num_mc_samples):
            l = torch.randint(0, num_decoders, (1,)).item()
            k = torch.randint(0, num_decoders, (1,)).item()
            f_l = decoders[l].decoder_net(curve[i].unsqueeze(0))     
            f_k = decoders[k].decoder_net(curve[i + 1].unsqueeze(0)) 

            diff = (f_l - f_k).view(-1) 
            step_energy = step_energy + (diff ** 2).sum()

        energy = energy + step_energy / num_mc_samples

    return energy

# functional codes innit

def curve_length_in_latent(curve):
    diffs = curve[1:] - curve[:-1]              
    lengths = diffs.norm(dim=-1)           
    return lengths.sum().item()

def encode_dataset(encoder, data_loader, device):
    # AI helped fixing this code.
    all_z, all_labels = [], []
    encoder.eval()
    with torch.no_grad():
        for x, y in data_loader:
            z = encoder(x.to(device)).mean.cpu()
            all_z.append(z)
            all_labels.append(y)
    return torch.cat(all_z).numpy(), torch.cat(all_labels).numpy()


def plot_latent_with_geodesics(all_z, all_labels, geodesics, title, save_path=None):

    colors = ['#AF58BA', '#FF1F5B', '#00CD6C']
    class_names = ['Digit 0', 'Digit 1', 'Digit 2']

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(8, 7))
        for cls in range(3):
            mask = all_labels == cls
            ax.scatter(
                all_z[mask, 0], all_z[mask, 1],
                c=colors[cls], alpha=0.3, s=8, label=class_names[cls]
            )
        for curve in geodesics:
            c = curve.cpu().numpy()
            ax.plot(c[:, 0], c[:, 1], 'k-', linewidth=1.2, alpha=0.75)
            ax.plot(c[0, 0],  c[0, 1],  'ko', markersize=4)
            ax.plot(c[-1, 0], c[-1, 1], 'ko', markersize=4)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.legend(loc='upper right')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Saved at: {save_path}")
        plt.close()


def plot_cov(decoder_counts, mean_euc_cov, mean_geo_cov, save_path=None):
    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(decoder_counts, mean_euc_cov, 'o-', color='#AF58BA',
                linewidth=2, markersize=8, label='Euclidean distance')
        ax.plot(decoder_counts, mean_geo_cov, 's-', color='#FF1F5B',
                linewidth=2, markersize=8, label='Geodesic distance')

        ax.set_xlabel('Number of ensemble decoders', fontsize=12)
        ax.set_ylabel('Average CoV (std / mean)', fontsize=12)
        ax.set_title('Part B: Reliability of distances vs. ensemble size', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_xticks(decoder_counts)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved at: {save_path}")
        plt.close()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics",
                 "train_ensemble", "ensemble_geodesics", "compute_cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument("--experiment-folder", type=str, default="experiment")
    parser.add_argument("--samples", type=str, default="samples.png")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32, metavar="N")
    parser.add_argument("--epochs-per-decoder", type=int, default=50, metavar="N")
    parser.add_argument("--latent-dim", type=int, default=2, metavar="N")
    parser.add_argument("--num-decoders", type=int, default=3, metavar="N",
                        help="taille de l'ensemble pour train_ensemble / ensemble_geodesics")
    parser.add_argument("--num-reruns", type=int, default=10, metavar="N",
                        help="nb de ré-entraînements pour compute_cov")
    parser.add_argument("--num-curves", type=int, default=25, metavar="N",
                        help="nb de géodésiques à tracer")
    parser.add_argument("--num-t", type=int, default=20, metavar="N",
                        help="nb de points le long de chaque courbe")
    parser.add_argument("--geodesic-steps", type=int, default=300, metavar="N",
                        help="nb d'itérations d'optimisation par géodésique")
    parser.add_argument("--geodesic-lr", type=float, default=1e-2,
                        help="learning rate pour l'optimiseur de géodésique")
    parser.add_argument("--num-pairs-cov", type=int, default=10, metavar="N",
                        help="nb de paires de test pour compute_cov")

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(f"  {key} = {value}")

    device = args.device

    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3

    train_tensors = datasets.MNIST(
        "data/", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(train_tensors.data, train_tensors.targets,
                           num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets,
                          num_train_data, num_classes)

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    M = args.latent_dim

    def new_encoder():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )

    def new_decoder():
        return nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    os.makedirs(args.experiment_folder, exist_ok=True)

    if args.mode == "train":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")
        print(f"Modèle sauvegardé dans {args.experiment_folder}/model.pt")

    elif args.mode == "sample":
        model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt",
                                         map_location=device))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(torch.cat([data.cpu(), recon.cpu()], dim=0),
                       "reconstruction_means.png")

    elif args.mode == "eval":
        model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt",
                                         map_location=device))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                elbos.append(model.elbo(x.to(device)).item())
        print(f"Mean test ELBO: {np.mean(elbos):.3f}")

    elif args.mode == "geodesics":
        print("\n=== PARTIE A : Géodésiques pull-back ===")

        model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())).to(device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt",
                                         map_location=device))
        model.eval()

        print("Encodage des données de test...")
        all_z, all_labels = encode_dataset(model.encoder, mnist_test_loader, device)
        all_z_tensor = torch.tensor(all_z, device=device)

        torch.manual_seed(123)
        N = len(all_z_tensor)
        num_pairs = args.num_curves

        idx1 = torch.randint(0, N, (num_pairs,))
        idx2 = torch.randint(0, N, (num_pairs,))

        decoder = model.decoder

        def energy_fn_single(curve):
            return compute_curve_energy(curve, decoder)

        print(f"Calcul de {num_pairs} géodésiques...")
        geodesics = []
        for i in tqdm(range(num_pairs)):
            z_start = all_z_tensor[idx1[i]].detach()
            z_end   = all_z_tensor[idx2[i]].detach()

            curve = compute_geodesic(
                z_start, z_end,
                energy_fn=energy_fn_single,
                num_t=args.num_t,
                num_steps=args.geodesic_steps,
                lr=args.geodesic_lr
            )
            geodesics.append(curve)

        save_path = f"{args.experiment_folder}/geodesics_partA.png"
        plot_latent_with_geodesics(
            all_z, all_labels, geodesics,
            title="Part A: Pull-back Geodesics (single decoder)",
            save_path=save_path
        )
        print("Partie A terminée !")

    elif args.mode == "train_ensemble":
        print(f"\n=== Part B : EnsembleVAE with ({args.num_decoders} decoders) ===")

        train_ensemble_vae(
            GaussianPrior(M), 
            new_encoder, 
            new_decoder, 
            args.num_decoders,
            mnist_train_loader, 
            args.epochs_per_decoder, 
            D = 20, 
            device = device, 
            lrate = 1e-3,
            folder = args.experiment_folder)
        print(f"EnsembleVAE finished with {20} retrainings with lr = {1e-3} and {args.num_decoders} decoders.")

    elif args.mode == "ensemble_geodesics":
        model = EnsembleVAE(
            prior=GaussianPrior(M),
            encoder=GaussianEncoder(new_encoder()),
            decoders=[GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
        ).to(device)
        model_path = f"{args.experiment_folder}/model_3dec_3.pt" # maybe change to 13 (?)
        print("The model is", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_z, all_labels = encode_dataset(model.encoder, mnist_test_loader, device)
        all_z_tensor = torch.tensor(all_z, device=device)

        torch.manual_seed(123)
        N = len(all_z_tensor)
        num_pairs = args.num_curves
        idx1 = torch.randint(0, N, (num_pairs,))
        idx2 = torch.randint(0, N, (num_pairs,))

        decoders = list(model.decoders)

        def energy_fn_ensemble(curve):
            return compute_ensemble_energy(curve, decoders, num_mc_samples=10)

        print(f"Calcul de {num_pairs} géodésiques (ensemble)...:)")
        geodesics = []
        for i in tqdm(range(num_pairs)):
            z_start = all_z_tensor[idx1[i]].detach()
            z_end   = all_z_tensor[idx2[i]].detach()

            curve = compute_geodesic(
                z_start, z_end,
                energy_fn=energy_fn_ensemble,
                num_t=args.num_t,
                num_steps=args.geodesic_steps,
                lr=1e-2
            )
            geodesics.append(curve)

        # Trace et sauvegarde
        save_path = f"{args.experiment_folder}/geodesics_partB_{args.num_decoders}dec.png"
        plot_latent_with_geodesics(
            all_z, all_labels, geodesics,
            title=f"Part B: Ensemble Geodesics ({args.num_decoders} decoders)",
            save_path=save_path
                            )
        torch.save(geodesics, f"{args.experiment_folder}/geodesic_factosmactos.pt")
        print("Partie B (geodésiques) terminée !")

    elif args.mode == "compute_cov":

        print("\n=== Part B : Calculating the CoV ===")
        test_x_list = []
        for x, _ in mnist_test_loader:
            test_x_list.append(x)
            #test_y_list.append(y)
        test_x = torch.cat(test_x_list) 

        num_pairs = args.num_pairs_cov  
        torch.manual_seed(123)  
        perm1 = torch.randperm(len(test_x))[:num_pairs]
        perm2 = torch.randperm(len(test_x))[:num_pairs]
        x_pairs_1 = test_x[perm1]   
        x_pairs_2 = test_x[perm2]

        decoder_counts = [1, 2, 3]

        euclidean_dists = torch.zeros(len(decoder_counts), args.num_reruns, num_pairs)
        geodesic_dists  = torch.zeros(len(decoder_counts), args.num_reruns, num_pairs)

        for run in range(args.num_reruns):
            print(f"\n{'='*50}")
            print(f"Initiating model {run+1}/{args.num_reruns}")
            print(f"\n{'='*50}")

            model = EnsembleVAE(
                        prior=GaussianPrior(M),
                        encoder=GaussianEncoder(new_encoder()),
                        decoders=[GaussianDecoder(new_decoder()) for _ in range(3)]
                    ).to(device)
            model_path = f"{args.experiment_folder}/3dec/model_3dec_{run}.pt"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            for dc_idx, num_dec in enumerate(decoder_counts):
                print(f"\n{'='*50}")
                print(f"  Configuration : {num_dec} décodeur(s)")
                print(f"{'='*50}")
                decoders = list(model.decoders)[:num_dec]
    
                with torch.no_grad(): # x_i^m, x_j^m
                    z1 = model.encoder(x_pairs_1.to(device)).mean 
                    z2 = model.encoder(x_pairs_2.to(device)).mean  

                euc = (z1 - z2).norm(dim=-1)  # (num_pairs,)
                euclidean_dists[dc_idx, run] = euc.detach().cpu()
                for p in tqdm(range(num_pairs), desc=f"Run {run+1} geodesics"):
                    z_start = z1[p].detach()
                    z_end   = z2[p].detach()

                    if num_dec == 1:
                        ef = lambda c, d = decoders: compute_curve_energy(c, d[0])
                    else:
                        ef = lambda c, d = decoders: compute_ensemble_energy(c, d, num_mc_samples=5)

                    curve = compute_geodesic(
                        z_start, z_end,
                        energy_fn=ef,
                        num_t=args.num_t,
                        num_steps=args.geodesic_steps,
                        lr=0.01
                    )
                    geodesic_dists[dc_idx, run, p] = curve_length_in_latent(curve)

                    # print(f"    Euclidean mean: {euclidean_dists[dc_idx, run].mean():.4f}")
                    # print(f"    Geodesic  mean: {geodesic_dists[dc_idx, run].mean():.4f}")

       
        euc_cov = (euclidean_dists.std(dim=1)
                   / (euclidean_dists.mean(dim=1))) #incase of division by zero
        geo_cov = (geodesic_dists.std(dim=1)
                   / (geodesic_dists.mean(dim=1))) #incase of division by zero

        mean_euc_cov = euc_cov.mean(dim=1).numpy()
        mean_geo_cov = geo_cov.mean(dim=1).numpy()

        print("\n=== CoV results ===")
        for i, nd in enumerate(decoder_counts):
            print(f"{nd} decoders(s): "
                  f"Euclidean CoV ={mean_euc_cov[i]:.4f}, "
                  f"Geodesic CoV ={mean_geo_cov[i]:.4f}")


        cov_path = f"{args.experiment_folder}/cov_plot.png"
        plot_cov(decoder_counts, mean_euc_cov, mean_geo_cov, save_path=cov_path)

        # Sauvegarde aussi les données brutes
        np.save(f"{args.experiment_folder}/euclidean_dists.npy",
                euclidean_dists.numpy())
        np.save(f"{args.experiment_folder}/geodesic_dists.npy",
                geodesic_dists.numpy())

        print("\nPart B (CoV) finished!!! wow")
        print(f"  Graph : {cov_path}")