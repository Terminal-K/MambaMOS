weight = 'exp/semantic_kitti/mambamos/model/model_best.pth'
resume = False
evaluate = False
test_only = False
seed = 1024
save_path = 'exp/semantic_kitti/mambamos'
num_worker = 24
batch_size = 4
batch_size_val = None
batch_size_test = None
epoch = 50
eval_epoch = 50
sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False
mix_prob = 0.0
param_dicts = [dict(keyword='block', lr=3.2000000000000005e-05)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='CheckpointSaver', save_freq=1)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
train_split = 'train'
val_split = 'val'
test_split = 'test'
ignore_index = 0
num_classes = 4
names = ['unlabeled', 'static', 'movable', 'moving']
lr_per_sample = 8e-05
data_root = 'data/semantic_kitti'
gride_size = 0.09
gather_num = 8
shuffle_orders = False
scan_modulation = False
dataset_type = 'SemanticKITTIMultiScansDataset'
model = dict(
    type='DefaultSegmentorV2',
    num_classes=4,
    backbone_out_channels=64,
    backbone=dict(
        type="MambaMOS",
        in_channels=5,  # x, y, z, i, t
        gather_num=gather_num,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=shuffle_orders,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=0),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=0)
    ])
optimizer = dict(type='AdamW', lr=0.00032, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.00032, 3.2000000000000005e-05],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
data = dict(
    num_classes=4,
    ignore_index=0,
    names=['unlabeled', 'static', 'movable', 'moving'],
    train=dict(
        type='SemanticKITTIMultiScansDataset',
        split='train',
        data_root='data/semantic_kitti',
        gather_num=8,
        scan_modulation=False,
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.09,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment', 'tn'),
                return_grid_coord=True),
            dict(type='SphereCrop', sample_rate=0.8, mode='random'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'tn'),
                feat_keys=('coord', 'strength', 'tn'))
        ],
        test_mode=False,
        ignore_index=0,
        loop=1),
    val=dict(
        type='SemanticKITTIMultiScansDataset',
        split='val',
        data_root='data/semantic_kitti',
        gather_num=8,
        scan_modulation=False,
        transform=[
            dict(
                type='GridSample',
                grid_size=0.09,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment', 'tn'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'tn'),
                feat_keys=('coord', 'strength', 'tn'))
        ],
        test_mode=False,
        ignore_index=0),
    test=dict(
        type='SemanticKITTIMultiScansDataset',
        split='val',
        data_root='data/semantic_kitti',
        gather_num=8,
        transform=[
            dict(
                type='Copy',
                keys_dict=dict(segment='origin_segment', tn='origin_tn')),
            dict(
                type='GridSample',
                grid_size=0.045,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment', 'tn'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            scale2kitti=None,
            voxelize=dict(
                type='GridSample',
                grid_size=0.09,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'strength', 'tn')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index', 'tn'),
                    feat_keys=('coord', 'strength', 'tn'))
            ],
            aug_transform=[[{
                'type': 'Add'
            }]]),
        ignore_index=0))
