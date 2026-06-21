"""
Tests for IncomeNN, VAE, and vae_loss in main.py.
All tests use random weights — no training required.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from main import IncomeNN, VAE, vae_loss


INPUT_DIM = 14
LATENT_DIM = 4
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def income_nn():
    model = IncomeNN(INPUT_DIM)
    model.eval()
    return model


@pytest.fixture(scope="module")
def vae():
    model = VAE(INPUT_DIM, latent_dim=LATENT_DIM)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# IncomeNN tests
# ---------------------------------------------------------------------------

def test_income_nn_output_shape(income_nn):
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    out = income_nn(x)
    assert out.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {out.shape}"


def test_income_nn_output_in_01(income_nn):
    x = torch.randn(100, INPUT_DIM)
    out = income_nn(x)
    assert out.min().item() >= 0.0, "NN output below 0 (expected sigmoid)"
    assert out.max().item() <= 1.0, "NN output above 1 (expected sigmoid)"


def test_income_nn_single_instance(income_nn):
    x = torch.randn(1, INPUT_DIM)
    out = income_nn(x)
    assert out.shape == (1,)


def test_income_nn_backward_works():
    """Gradient flow must work — required for gradient CF method."""
    model = IncomeNN(INPUT_DIM)
    model.train()
    x = torch.randn(4, INPUT_DIM, requires_grad=False)
    x_cf = x.clone().detach().requires_grad_(True)
    out = model(x_cf)
    loss = F.binary_cross_entropy(out, torch.zeros(4))
    loss.backward()
    assert x_cf.grad is not None, "Gradient did not flow to input"
    assert not torch.isnan(x_cf.grad).any(), "NaN gradient detected"


def test_income_nn_no_inplace_ops_on_input():
    """In-place ops on input would break autograd — verify input unchanged after forward."""
    model = IncomeNN(INPUT_DIM)
    x = torch.randn(4, INPUT_DIM)
    x_before = x.clone()
    _ = model(x)
    assert torch.allclose(x, x_before), "Forward pass modified the input tensor in-place"


# ---------------------------------------------------------------------------
# VAE tests
# ---------------------------------------------------------------------------

def test_vae_forward_shapes(vae):
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    x_recon, mu, logvar = vae(x)
    assert x_recon.shape == (BATCH_SIZE, INPUT_DIM)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, LATENT_DIM)


def test_vae_reparameterize_shape(vae):
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
    z = vae.reparameterize(mu, logvar)
    assert z.shape == (BATCH_SIZE, LATENT_DIM)


def test_vae_encode_returns_mu_logvar(vae):
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    mu, logvar = vae.encode(x)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM)
    assert logvar.shape == (BATCH_SIZE, LATENT_DIM)


def test_vae_decode_shape(vae):
    z = torch.randn(BATCH_SIZE, LATENT_DIM)
    out = vae.decode(z)
    assert out.shape == (BATCH_SIZE, INPUT_DIM)


# ---------------------------------------------------------------------------
# vae_loss tests
# ---------------------------------------------------------------------------

def test_vae_loss_is_scalar():
    x_recon = torch.randn(BATCH_SIZE, INPUT_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    mu = torch.randn(BATCH_SIZE, LATENT_DIM)
    logvar = torch.randn(BATCH_SIZE, LATENT_DIM)
    loss = vae_loss(x_recon, x, mu, logvar)
    assert loss.shape == torch.Size([])


def test_vae_loss_nonnegative_for_typical_inputs():
    """ELBO loss can be negative in theory but is typically positive for random inits."""
    x_recon = torch.randn(BATCH_SIZE, INPUT_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)
    loss = vae_loss(x_recon, x, mu, logvar)
    assert loss.item() > 0.0


def test_vae_loss_zero_kl_when_standard_normal():
    """
    When mu=0 and logvar=0 (i.e., q = N(0,1)):
    KL(N(0,1) || N(0,1)) = 0, so loss = reconstruction MSE only.
    """
    x_recon = torch.randn(BATCH_SIZE, INPUT_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    logvar = torch.zeros(BATCH_SIZE, LATENT_DIM)  # log(var=1) = 0
    loss = vae_loss(x_recon, x, mu, logvar, beta=1.0)
    expected_recon = F.mse_loss(x_recon, x, reduction="sum")
    assert abs(loss.item() - expected_recon.item()) < 1e-3
