custom_imports = dict(
    imports=['helix.integrations.pimm'], allow_failed_imports=False)
CORPUS = '/sdf/data/neutrino/omara/coeff_tpc/run_0027575715'
CKPT = '/sdf/data/neutrino/omara/archive/fm_m113_converted.pt'
CELL_T = 'grid_center'
BINS = '/sdf/data/neutrino/omara/archive/coeff_bins_run0027575715.pt'
weight = None
resume = False
evaluate = True
test_only = False
seed = 0
save_path = 'exp/coeff_fm_encode'
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4
epoch = 1
eval_epoch = 1
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'bfloat16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
matmul_precision = 'high'
prefetch_factor = None
detect_anomaly = False
mix_prob = 0
deterministic = False
param_dicts = None
structured_logging = dict(
    enabled=False,
    trace_hooks=False,
    batch_stats_every=1,
    max_file_size_mb=128,
    backup_count=3)
model = dict(
    type='Coeff-FM',
    checkpoint='/sdf/data/neutrino/omara/archive/fm_m113_converted.pt',
    weights=True,
    bins='/sdf/data/neutrino/omara/archive/coeff_bins_run0027575715.pt')
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0003,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
transform = [
    dict(
        type='CoeffTokenize',
        part='coeff',
        clean_part='coeff_clean',
        fm_names=True,
        cfg=PatchConfig(
            pw=16,
            pt=8,
            n_bands=4,
            lev=(4, 4, 3, 2),
            delta=(-2.38, 0.62, 0.75, 0.5),
            toff=(-17.4, 2.6, 5.5),
            cell_t='grid_center',
            sigma_norm=2.6)),
    dict(type='CoeffCollect', part='coeff')
]
_data_common = dict(
    type='CoeffTPCDataset',
    data_root='/sdf/data/neutrino/omara/coeff_tpc/run_0027575715',
    dataset_name='sim_wire',
    modalities=('coeff', 'coeff_clean'),
    transform=[
        dict(
            type='CoeffTokenize',
            part='coeff',
            clean_part='coeff_clean',
            fm_names=True,
            cfg=PatchConfig(
                pw=16,
                pt=8,
                n_bands=4,
                lev=(4, 4, 3, 2),
                delta=(-2.38, 0.62, 0.75, 0.5),
                toff=(-17.4, 2.6, 5.5),
                cell_t='grid_center',
                sigma_norm=2.6)),
        dict(type='CoeffCollect', part='coeff')
    ])
data = dict(
    train=dict(
        type='CoeffTPCDataset',
        data_root='/sdf/data/neutrino/omara/coeff_tpc/run_0027575715',
        dataset_name='sim_wire',
        modalities=('coeff', 'coeff_clean'),
        transform=[
            dict(
                type='CoeffTokenize',
                part='coeff',
                clean_part='coeff_clean',
                fm_names=True,
                cfg=PatchConfig(
                    pw=16,
                    pt=8,
                    n_bands=4,
                    lev=(4, 4, 3, 2),
                    delta=(-2.38, 0.62, 0.75, 0.5),
                    toff=(-17.4, 2.6, 5.5),
                    cell_t='grid_center',
                    sigma_norm=2.6)),
            dict(type='CoeffCollect', part='coeff')
        ]),
    val=dict(
        type='CoeffTPCDataset',
        data_root='/sdf/data/neutrino/omara/coeff_tpc/run_0027575715',
        dataset_name='sim_wire',
        modalities=('coeff', 'coeff_clean'),
        transform=[
            dict(
                type='CoeffTokenize',
                part='coeff',
                clean_part='coeff_clean',
                fm_names=True,
                cfg=PatchConfig(
                    pw=16,
                    pt=8,
                    n_bands=4,
                    lev=(4, 4, 3, 2),
                    delta=(-2.38, 0.62, 0.75, 0.5),
                    toff=(-17.4, 2.6, 5.5),
                    cell_t='grid_center',
                    sigma_norm=2.6)),
            dict(type='CoeffCollect', part='coeff')
        ]),
    test=dict(
        type='CoeffTPCDataset',
        data_root='/sdf/data/neutrino/omara/coeff_tpc/run_0027575715',
        dataset_name='sim_wire',
        modalities=('coeff', 'coeff_clean'),
        transform=[
            dict(
                type='CoeffTokenize',
                part='coeff',
                clean_part='coeff_clean',
                fm_names=True,
                cfg=PatchConfig(
                    pw=16,
                    pt=8,
                    n_bands=4,
                    lev=(4, 4, 3, 2),
                    delta=(-2.38, 0.62, 0.75, 0.5),
                    toff=(-17.4, 2.6, 5.5),
                    cell_t='grid_center',
                    sigma_norm=2.6)),
            dict(type='CoeffCollect', part='coeff')
        ]))
hooks = [
    dict(type='HelixPathBootstrap'),
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='CoeffFMEvaluator', max_batches=32),
    dict(type='CheckpointSaver', save_freq=None)
]
train = dict(type='FMTrainer')
