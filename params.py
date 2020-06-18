from types import SimpleNamespace

data = {
    'NUM_AGENTS': 2,
    'HORIZON': 128,
    'GAMMA': 0.99,

    'ENT_MAX': 0.00,
    'ENT_MIN': 0.01,
    'ENT_STEP': 5e+4,
    'LR': 1e-3,

    # test
    'TEST_ITER': 20,

    # Train status
    'NUM_UPDATE': 0,
    'NUM_EPISODE': 0,
}

def get():
    params = SimpleNamespace(**data)
    return params