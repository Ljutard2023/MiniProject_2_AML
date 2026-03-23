# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024
# Geodesics + Ensemble completed by students, 2026

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

# ===========================================================================
# EXISTING CLASSES (provided)
# ===========================================================================

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


# ===========================================================================
# NEW — PARTIE A : MÉTRIQUES PULL-BACK ET GÉODÉSIQUES
# ===========================================================================

def compute_curve_energy(curve, decoder):
    """
    Calcule l'énergie d'une courbe sous la métrique pull-back.

    L'idée : au lieu de mesurer les distances dans l'espace latent 2D,
    on "pousse" la courbe à travers le décodeur et on mesure les distances
    dans l'espace des images (28×28). Cela donne une métrique qui respecte
    la géométrie apprise par le VAE.

    Formule :
        E(c) ≈ Σ_i ||f(c(t_{i+1})) - f(c(t_i))||²

    où f est la moyenne du décodeur gaussien.

    Paramètres:
    -----------
    curve   : torch.Tensor, forme (T, M)
              T points consécutifs dans l'espace latent
    decoder : GaussianDecoder
              Le décodeur du VAE

    Retourne:
    ---------
    energy  : torch.Tensor (scalaire)
    """
    # Passe tous les points de la courbe dans le décodeur → (T, 1, 28, 28)
    means = decoder.decoder_net(curve)

    # Aplatit les dimensions spatiales → (T, 784)
    means_flat = means.view(means.shape[0], -1)

    # Différences entre points consécutifs décodés → (T-1, 784)
    diffs = means_flat[1:] - means_flat[:-1]

    # Somme des normes au carré
    energy = (diffs ** 2).sum()
    return energy


def compute_ensemble_energy(curve, decoders, num_mc_samples=5):
    """
    Calcule l'énergie pull-back moyennée sur l'ensemble (Équation 1 du sujet).

    Pour chaque pas i de la courbe, on tire aléatoirement deux décodeurs
    l et k, et on calcule ||f_l(c(t_i)) - f_k(c(t_{i+1}))||².
    On répète num_mc_samples fois (Monte Carlo) pour réduire la variance.

    Formule :
        E(c) ≈ Σ_i E_{l,k}[ ||f_l(c(t_i)) - f_k(c(t_{i+1}))||² ]

    Paramètres:
    -----------
    curve          : torch.Tensor, forme (T, M)
    decoders       : list[GaussianDecoder]  — les membres de l'ensemble
    num_mc_samples : int — nombre d'échantillons Monte Carlo par pas

    Retourne:
    ---------
    energy : torch.Tensor (scalaire)
    """
    T = curve.shape[0]
    num_decoders = len(decoders)
    energy = torch.tensor(0.0, device=curve.device)

    for i in range(T - 1):
        step_energy = torch.tensor(0.0, device=curve.device)

        for _ in range(num_mc_samples):
            # Tire l et k uniformément parmi les décodeurs disponibles
            l = torch.randint(0, num_decoders, (1,)).item()
            k = torch.randint(0, num_decoders, (1,)).item()

            # Évalue les deux décodeurs aux points consécutifs
            f_l = decoders[l].decoder_net(curve[i].unsqueeze(0))      # (1, 1, 28, 28)
            f_k = decoders[k].decoder_net(curve[i + 1].unsqueeze(0))  # (1, 1, 28, 28)

            diff = (f_l - f_k).view(-1)  # aplatit
            step_energy = step_energy + (diff ** 2).sum()

        # Moyenne sur les échantillons MC
        energy = energy + step_energy / num_mc_samples

    return energy


def compute_geodesic(z_start, z_end, energy_fn, num_t=20, num_steps=300, lr=1e-2):
    """
    Calcule une géodésique entre deux points latents en minimisant l'énergie.

    Principe :
    - On initialise la courbe comme une ligne droite entre z_start et z_end
    - On fixe les extrémités (elles ne bougent pas)
    - On optimise les points intérieurs avec Adam pour minimiser l'énergie

    Paramètres:
    -----------
    z_start    : torch.Tensor, forme (M,)  — point de départ dans l'espace latent
    z_end      : torch.Tensor, forme (M,)  — point d'arrivée
    energy_fn  : callable (curve: Tensor) -> scalar Tensor
                 Fonction qui calcule l'énergie d'une courbe
    num_t      : int — nombre de points sur la courbe (résolution)
    num_steps  : int — nombre d'itérations d'optimisation
    lr         : float — learning rate pour Adam

    Retourne:
    ---------
    curve : torch.Tensor, forme (num_t, M) — la géodésique optimisée
    """
    device = z_start.device

    # ---- Initialisation : ligne droite entre les deux points ----
    # t_vals ∈ [0, 1] répartis uniformément
    t_vals = torch.linspace(0, 1, num_t, device=device)
    with torch.no_grad():
        curve_init = torch.stack([(1 - t) * z_start + t * z_end for t in t_vals])
        # forme : (num_t, M)

    # ---- Paramètres à optimiser : UNIQUEMENT les points intérieurs ----
    # Les points [0] et [-1] sont fixes (= z_start et z_end)
    inner = curve_init[1:-1].clone().detach().requires_grad_(True)
    # forme : (num_t - 2, M)

    optimizer = torch.optim.Adam([inner], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()

        # Reconstruit la courbe complète avec les extrémités fixes
        curve = torch.cat([
            z_start.unsqueeze(0),   # (1, M) — fixe
            inner,                  # (num_t-2, M) — appris
            z_end.unsqueeze(0)      # (1, M) — fixe
        ], dim=0)

        # Calcule et rétropropage l'énergie
        energy = energy_fn(curve)
        energy.backward()
        optimizer.step()

    # Retourne la courbe finale détachée du graphe de calcul
    with torch.no_grad():
        curve = torch.cat([
            z_start.unsqueeze(0),
            inner.detach(),
            z_end.unsqueeze(0)
        ], dim=0)

    return curve


def curve_length_in_latent(curve):
    """
    Calcule la longueur euclidienne d'une courbe dans l'espace latent.
    Utilisée comme proxy de la 'distance géodésique'.

    Paramètres:
    -----------
    curve : torch.Tensor, forme (T, M)

    Retourne:
    ---------
    length : float
    """
    diffs = curve[1:] - curve[:-1]              # (T-1, M)
    lengths = diffs.norm(dim=-1)                 # (T-1,)
    return lengths.sum().item()


# ===========================================================================
# NEW — PARTIE B : ENSEMBLE VAE
# ===========================================================================

class EnsembleVAE(nn.Module):
    """
    VAE avec un ensemble de décodeurs partageant un seul encodeur.

    Architecture :
        - 1 encodeur commun
        - N décodeurs indépendants

    Pourquoi ? Plusieurs décodeurs = plusieurs "visions" de l'espace de données.
    En moyennant leur énergie, on rend les géodésiques plus robustes aux
    variations d'entraînement.
    """

    def __init__(self, prior, encoder, decoders):
        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)  # liste de GaussianDecoder

    def elbo(self, x, decoder_idx=0):
        """Calcule l'ELBO en utilisant le décodeur numéro decoder_idx."""
        q = self.encoder(x)
        z = q.rsample()
        decoder = self.decoders[decoder_idx]
        return torch.mean(
            decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )

    def forward(self, x, decoder_idx=0):
        return -self.elbo(x, decoder_idx)


def train_ensemble_vae(new_encoder_fn, new_decoder_fn, M, num_decoders,
                       data_loader, epochs_per_decoder, device):
    """
    Entraîne un EnsembleVAE avec la stratégie suivante :

    1. Entraîne l'encodeur ET le 1er décodeur ensemble normalement.
    2. Gèle l'encodeur (ses poids ne changent plus).
    3. Pour chaque décodeur supplémentaire : entraîne-le seul avec l'encodeur gelé.

    Cette stratégie garantit que tous les décodeurs partagent le même espace
    latent (défini par l'encodeur du 1er entraînement), mais apprennent des
    reconstructions différentes.

    Paramètres:
    -----------
    new_encoder_fn    : callable → nn.Module  (fabrique un nouvel encodeur)
    new_decoder_fn    : callable → nn.Module  (fabrique un nouveau décodeur)
    M                 : int — dimension de l'espace latent
    num_decoders      : int — nombre de décodeurs dans l'ensemble
    data_loader       : DataLoader
    epochs_per_decoder: int — epochs d'entraînement par décodeur
    device            : str

    Retourne:
    ---------
    model : EnsembleVAE entraîné
    """

    def noise(x, std=0.05):
        return torch.clamp(x + std * torch.randn_like(x), 0.0, 1.0)

    # Crée le modèle avec tous les décodeurs
    model = EnsembleVAE(
        prior=GaussianPrior(M),
        encoder=GaussianEncoder(new_encoder_fn()),
        decoders=[GaussianDecoder(new_decoder_fn()) for _ in range(num_decoders)]
    ).to(device)

    # ---- Étape 1 : entraîne encodeur + décodeur 0 ensemble ----
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoders[0].parameters()),
        lr=1e-3
    )
    num_steps = len(data_loader) * epochs_per_decoder
    with tqdm(range(num_steps), desc=f"[Decoder 1/{num_decoders}]") as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0].to(device)
                x = noise(x)
                optimizer.zero_grad()
                loss = model(x, decoder_idx=0)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    pbar.set_description(
                        f"[Decoder 1/{num_decoders}] loss={loss.item():.1f}"
                    )
            except KeyboardInterrupt:
                break

    # ---- Étape 2 : gèle l'encodeur ----
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ---- Étape 3 : entraîne chaque décodeur supplémentaire ----
    for d_idx in range(1, num_decoders):
        optimizer = torch.optim.Adam(
            model.decoders[d_idx].parameters(), lr=1e-3
        )
        num_steps = len(data_loader) * epochs_per_decoder
        with tqdm(range(num_steps), desc=f"[Decoder {d_idx+1}/{num_decoders}]") as pbar:
            for step in pbar:
                try:
                    x = next(iter(data_loader))[0].to(device)
                    x = noise(x)
                    optimizer.zero_grad()
                    loss = model(x, decoder_idx=d_idx)
                    loss.backward()
                    optimizer.step()
                    if step % 10 == 0:
                        pbar.set_description(
                            f"[Decoder {d_idx+1}/{num_decoders}] loss={loss.item():.1f}"
                        )
                except KeyboardInterrupt:
                    break

    # Ré-active les gradients de l'encodeur (bonne pratique)
    for param in model.encoder.parameters():
        param.requires_grad = True

    return model


# ===========================================================================
# FONCTIONS UTILITAIRES DE TRACÉ
# ===========================================================================

def encode_dataset(encoder, data_loader, device):
    """
    Encode tous les points du dataset et retourne les moyennes latentes + labels.

    Retourne:
    ---------
    all_z      : np.ndarray, forme (N, M)
    all_labels : np.ndarray, forme (N,)
    """
    all_z, all_labels = [], []
    encoder.eval()
    with torch.no_grad():
        for x, y in data_loader:
            z = encoder(x.to(device)).mean.cpu()
            all_z.append(z)
            all_labels.append(y)
    return torch.cat(all_z).numpy(), torch.cat(all_labels).numpy()


def plot_latent_with_geodesics(all_z, all_labels, geodesics, title, save_path=None):
    """
    Trace l'espace latent 2D avec les classes colorées et les géodésiques.

    Paramètres:
    -----------
    all_z      : np.ndarray (N, 2) — coordonnées latentes de tous les points
    all_labels : np.ndarray (N,)   — labels des classes (0, 1, 2)
    geodesics  : list of torch.Tensor (T, 2) — courbes géodésiques
    title      : str
    save_path  : str ou None
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    class_names = ['Digit 0', 'Digit 1', 'Digit 2']

    # Trace les points latents colorés par classe
    for cls in range(3):
        mask = all_labels == cls
        ax.scatter(
            all_z[mask, 0], all_z[mask, 1],
            c=colors[cls], alpha=0.3, s=8, label=class_names[cls]
        )

    # Trace les géodésiques
    for curve in geodesics:
        c = curve.cpu().numpy()
        ax.plot(c[:, 0], c[:, 1], 'k-', linewidth=1.2, alpha=0.75)
        # Marque les extrémités
        ax.plot(c[0, 0],  c[0, 1],  'ko', markersize=4)
        ax.plot(c[-1, 0], c[-1, 1], 'ko', markersize=4)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Sauvegardé : {save_path}")
    plt.close()


def plot_cov(decoder_counts, mean_euc_cov, mean_geo_cov, save_path=None):
    """
    Trace le CoV moyen (Euclidien vs Géodésique) en fonction du nb de décodeurs.

    Paramètres:
    -----------
    decoder_counts : list[int]   — ex: [1, 2, 3]
    mean_euc_cov   : np.ndarray  — CoV euclidien moyen pour chaque nb de décodeurs
    mean_geo_cov   : np.ndarray  — CoV géodésique moyen
    save_path      : str ou None
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(decoder_counts, mean_euc_cov, 'o-', color='tab:blue',
            linewidth=2, markersize=8, label='Euclidean distance')
    ax.plot(decoder_counts, mean_geo_cov, 's-', color='tab:orange',
            linewidth=2, markersize=8, label='Geodesic distance')

    ax.set_xlabel('Number of ensemble decoders', fontsize=12)
    ax.set_ylabel('Average CoV (std / mean)', fontsize=12)
    ax.set_title('Part B: Reliability of distances vs. ensemble size', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(decoder_counts)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  → Sauvegardé : {save_path}")
    plt.close()


# ===========================================================================
# MAIN
# ===========================================================================

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

    # ------------------------------------------------------------------
    # Chargement des données MNIST (sous-ensemble 3 classes, 2048 points)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fabriques de réseaux (appelées pour créer de nouveaux réseaux)
    # ------------------------------------------------------------------
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

    # ==================================================================
    # MODE : train  (inchangé)
    # ==================================================================
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

    # ==================================================================
    # MODE : sample  (inchangé)
    # ==================================================================
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

    # ==================================================================
    # MODE : eval  (inchangé)
    # ==================================================================
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

    # ==================================================================
    # MODE : geodesics  ← PARTIE A (nouveau code)
    # ==================================================================
    elif args.mode == "geodesics":
        """
        PARTIE A — Géodésiques sous la métrique pull-back d'un seul VAE.

        Étapes :
        1. Charge le VAE entraîné
        2. Encode les données de test → points dans l'espace latent 2D
        3. Choisit 25 paires aléatoires de points
        4. Pour chaque paire, optimise une courbe pour minimiser l'énergie pull-back
        5. Trace l'espace latent avec toutes les géodésiques
        """
        print("\n=== PARTIE A : Géodésiques pull-back ===")

        # Charge le modèle
        model = VAE(GaussianPrior(M), GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())).to(device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt",
                                         map_location=device))
        model.eval()

        # Encode toutes les données de test
        print("Encodage des données de test...")
        all_z, all_labels = encode_dataset(model.encoder, mnist_test_loader, device)
        all_z_tensor = torch.tensor(all_z, device=device)

        # Sélectionne des paires aléatoires reproductibles
        torch.manual_seed(42)
        N = len(all_z_tensor)
        num_pairs = args.num_curves  # ≥ 25 recommandé par le sujet

        idx1 = torch.randint(0, N, (num_pairs,))
        idx2 = torch.randint(0, N, (num_pairs,))

        # Définit la fonction d'énergie (utilise le seul décodeur du VAE)
        decoder = model.decoder

        def energy_fn_single(curve):
            return compute_curve_energy(curve, decoder)

        # Calcule les géodésiques
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

        # Trace et sauvegarde
        save_path = f"{args.experiment_folder}/geodesics_partA.png"
        plot_latent_with_geodesics(
            all_z, all_labels, geodesics,
            title="Part A: Pull-back Geodesics (single decoder)",
            save_path=save_path
        )
        print("Partie A terminée !")

    # ==================================================================
    # MODE : train_ensemble  ← PARTIE B, étape 1 (nouveau code)
    # ==================================================================
    elif args.mode == "train_ensemble":
        """
        PARTIE B — Entraîne un EnsembleVAE et le sauvegarde.

        Utilisez --num-decoders pour choisir le nombre de décodeurs.
        Le modèle est sauvegardé dans experiment_folder/ensemble_model.pt
        """
        print(f"\n=== PARTIE B : Entraînement EnsembleVAE ({args.num_decoders} décodeurs) ===")

        model = train_ensemble_vae(
            new_encoder_fn=new_encoder,
            new_decoder_fn=new_decoder,
            M=M,
            num_decoders=args.num_decoders,
            data_loader=mnist_train_loader,
            epochs_per_decoder=args.epochs_per_decoder,
            device=device
        )

        save_path = f"{args.experiment_folder}/ensemble_model_{args.num_decoders}dec.pt"
        torch.save(model.state_dict(), save_path)
        print(f"EnsembleVAE sauvegardé : {save_path}")

    # ==================================================================
    # MODE : ensemble_geodesics  ← PARTIE B, visualisation (nouveau code)
    # ==================================================================
    elif args.mode == "ensemble_geodesics":
        """
        PARTIE B — Trace les géodésiques calculées avec l'ensemble de décodeurs.

        Les mêmes paires de points qu'en Partie A sont utilisées
        (même seed aléatoire) pour la comparaison.
        """
        print(f"\n=== PARTIE B : Géodésiques ensemble ({args.num_decoders} décodeurs) ===")

        # Charge l'ensemble VAE sauvegardé
        model = EnsembleVAE(
            prior=GaussianPrior(M),
            encoder=GaussianEncoder(new_encoder()),
            decoders=[GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
        ).to(device)
        model_path = f"{args.experiment_folder}/ensemble_model_{args.num_decoders}dec.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Encode les données de test
        print("Encodage des données de test...")
        all_z, all_labels = encode_dataset(model.encoder, mnist_test_loader, device)
        all_z_tensor = torch.tensor(all_z, device=device)

        # MÊMES paires qu'en Partie A (seed=42)
        torch.manual_seed(42)
        N = len(all_z_tensor)
        num_pairs = args.num_curves
        idx1 = torch.randint(0, N, (num_pairs,))
        idx2 = torch.randint(0, N, (num_pairs,))

        # Liste des décodeurs de l'ensemble
        decoders = list(model.decoders)

        # Fonction d'énergie : utilise tous les décodeurs de l'ensemble
        def energy_fn_ensemble(curve):
            return compute_ensemble_energy(curve, decoders, num_mc_samples=5)

        # Calcule les géodésiques avec l'ensemble
        print(f"Calcul de {num_pairs} géodésiques (ensemble)...")
        geodesics = []
        for i in tqdm(range(num_pairs)):
            z_start = all_z_tensor[idx1[i]].detach()
            z_end   = all_z_tensor[idx2[i]].detach()

            curve = compute_geodesic(
                z_start, z_end,
                energy_fn=energy_fn_ensemble,
                num_t=args.num_t,
                num_steps=args.geodesic_steps,
                lr=args.geodesic_lr
            )
            geodesics.append(curve)

        # Trace et sauvegarde
        save_path = f"{args.experiment_folder}/geodesics_partB_{args.num_decoders}dec.png"
        plot_latent_with_geodesics(
            all_z, all_labels, geodesics,
            title=f"Part B: Ensemble Geodesics ({args.num_decoders} decoders)",
            save_path=save_path
        )
        print("Partie B (geodésiques) terminée !")

    # ==================================================================
    # MODE : compute_cov  ← PARTIE B, analyse CoV (nouveau code)
    # ==================================================================
    elif args.mode == "compute_cov":
        """
        PARTIE B — Calcule et trace le Coefficient of Variation (CoV).

        Pour chaque nombre de décodeurs dans [1, 2, 3] :
          Pour chaque run parmi num_reruns ré-entraînements :
            1. Entraîne un EnsembleVAE
            2. Encode les paires de test fixes
            3. Calcule les distances euclidiennes et géodésiques

        Puis calcule CoV = std / mean à travers les runs, et trace la courbe.

        ATTENTION : Cette étape est LONGUE (plusieurs heures).
                    Commencez tôt !
        """
        print("\n=== PARTIE B : Calcul du CoV ===")
        print(f"  {args.num_reruns} runs × 3 configs = {args.num_reruns * 3} entraînements")
        print("  Cela peut prendre plusieurs heures...\n")

        # ------------------------------------------------------------------
        # 1. Fixe les paires de test (IDENTIQUES pour tous les modèles)
        # ------------------------------------------------------------------
        # Charge toutes les données de test en mémoire
        test_x_list, test_y_list = [], []
        for x, y in mnist_test_loader:
            test_x_list.append(x)
            test_y_list.append(y)
        test_x = torch.cat(test_x_list)  # (N_test, 1, 28, 28)

        num_pairs = args.num_pairs_cov  # ≥ 10 selon le sujet
        torch.manual_seed(0)  # seed fixe → mêmes paires pour tous les runs
        perm1 = torch.randperm(len(test_x))[:num_pairs]
        perm2 = torch.randperm(len(test_x))[:num_pairs]
        x_pairs_1 = test_x[perm1]   # (num_pairs, 1, 28, 28)
        x_pairs_2 = test_x[perm2]

        # ------------------------------------------------------------------
        # 2. Stocke les distances pour chaque config et chaque run
        # ------------------------------------------------------------------
        decoder_counts = [1, 2, 3]

        # Shape : (nb_configs, num_reruns, num_pairs)
        euclidean_dists = torch.zeros(len(decoder_counts), args.num_reruns, num_pairs)
        geodesic_dists  = torch.zeros(len(decoder_counts), args.num_reruns, num_pairs)

        for dc_idx, num_dec in enumerate(decoder_counts):
            print(f"\n{'='*50}")
            print(f"  Configuration : {num_dec} décodeur(s)")
            print(f"{'='*50}")

            for run in range(args.num_reruns):
                print(f"\n  Run {run+1}/{args.num_reruns}")

                # ---- Entraîne un nouveau EnsembleVAE ----
                model = train_ensemble_vae(
                    new_encoder_fn=new_encoder,
                    new_decoder_fn=new_decoder,
                    M=M,
                    num_decoders=num_dec,
                    data_loader=mnist_train_loader,
                    epochs_per_decoder=args.epochs_per_decoder,
                    device=device
                )
                model.eval()

                # ---- Encode les paires de test ----
                with torch.no_grad():
                    z1 = model.encoder(x_pairs_1.to(device)).mean  # (num_pairs, 2)
                    z2 = model.encoder(x_pairs_2.to(device)).mean  # (num_pairs, 2)

                # ---- Distances euclidiennes ----
                euc = (z1 - z2).norm(dim=-1)  # (num_pairs,)
                euclidean_dists[dc_idx, run] = euc.detach().cpu()

                # ---- Distances géodésiques ----
                decoders = list(model.decoders)

                for p in range(num_pairs):
                    z_start = z1[p].detach()
                    z_end   = z2[p].detach()

                    # Sélectionne la bonne fonction d'énergie
                    if num_dec == 1:
                        ef = lambda c: compute_curve_energy(c, decoders[0])
                    else:
                        ef = lambda c: compute_ensemble_energy(c, decoders, num_mc_samples=3)

                    curve = compute_geodesic(
                        z_start, z_end,
                        energy_fn=ef,
                        num_t=args.num_t,
                        num_steps=args.geodesic_steps,
                        lr=args.geodesic_lr
                    )
                    # Longueur de la géodésique dans l'espace latent
                    geodesic_dists[dc_idx, run, p] = curve_length_in_latent(curve)

                print(f"    Euclidean mean: {euclidean_dists[dc_idx, run].mean():.4f}")
                print(f"    Geodesic  mean: {geodesic_dists[dc_idx, run].mean():.4f}")

        # ------------------------------------------------------------------
        # 3. Calcule le CoV = std / mean  sur les runs, pour chaque paire
        # ------------------------------------------------------------------
        # CoV shape : (nb_configs, num_pairs)
        euc_cov = (euclidean_dists.std(dim=1)
                   / (euclidean_dists.mean(dim=1) + 1e-10))
        geo_cov = (geodesic_dists.std(dim=1)
                   / (geodesic_dists.mean(dim=1) + 1e-10))

        # Moyenne du CoV sur toutes les paires
        mean_euc_cov = euc_cov.mean(dim=1).numpy()  # (nb_configs,)
        mean_geo_cov = geo_cov.mean(dim=1).numpy()

        print("\n=== Résultats CoV ===")
        for i, nd in enumerate(decoder_counts):
            print(f"  {nd} décodeur(s) : "
                  f"CoV euclidien={mean_euc_cov[i]:.4f}, "
                  f"CoV géodésique={mean_geo_cov[i]:.4f}")

        # ------------------------------------------------------------------
        # 4. Trace et sauvegarde
        # ------------------------------------------------------------------
        cov_path = f"{args.experiment_folder}/cov_plot.png"
        plot_cov(decoder_counts, mean_euc_cov, mean_geo_cov, save_path=cov_path)

        # Sauvegarde aussi les données brutes
        np.save(f"{args.experiment_folder}/euclidean_dists.npy",
                euclidean_dists.numpy())
        np.save(f"{args.experiment_folder}/geodesic_dists.npy",
                geodesic_dists.numpy())

        print("\nPartie B (CoV) terminée !")
        print(f"  Graphique : {cov_path}")