from surepy.simulation import simulate_csr, simulate_blobs


def test_simulate_csr():
    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = None, z_range = None, seed=0)
    assert(len(dat) == 10)
    # dat.print_meta()

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = None, seed=0)
    assert(len(dat) == 10)

    dat = simulate_csr(n_samples = 10, x_range = (0,10000), y_range = (0,10000), z_range = (0,10000), seed=0)
    assert(len(dat) == 10)


def test_simulate_blobs_1D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=1, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert ('Position_x' in dat.data.columns)
    # dat.print_meta()

def test_simulate_blobs_2D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=2, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert ('Position_y' in dat.data.columns)

def test_simulate_blobs_3D():
    dat = simulate_blobs(n_centers=10, n_samples=100, n_features=3, center_box=(0, 10000), cluster_std=10, seed=None)
    assert (len(dat) == 100)
    assert ('Position_z' in dat.data.columns)


