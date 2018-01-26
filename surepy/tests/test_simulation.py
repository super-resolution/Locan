from surepy.simulation import simulate_csr, simulate_blobs


def test_simulate_csr():
    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = None, z_range = None, seed=0)
    assert(len(dat) == 10)

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=0)
    assert(len(dat) == 10)

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = (0,10000), seed=0)
    assert(len(dat) == 10)


def test_simulate_blobs():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=2, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)


