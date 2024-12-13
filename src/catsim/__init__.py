import taichi as ti

_RANDOM_SEED = 231827438

ti.init(
    arch=ti.cpu,
    default_fp=ti.f32,
    default_ip=ti.i32,
    random_seed=_RANDOM_SEED,
    debug=True,
)
