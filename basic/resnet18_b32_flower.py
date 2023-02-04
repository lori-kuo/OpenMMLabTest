_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']

model = dict(
    head=dict(
        num_classes=6,
        topk = (1,)
))

# 数据集地址和预训练模型地址是针对命令行环境的，如果cd到mmclassification，下面的就不用调整
data = dict(
    samples_per_gpu = 32,
    workers_per_gpu = 2,
    train = dict(
        data_prefix = 'data/train',
        ann_file = 'data/train.txt',
        classes = 'data/classes.txt'
    ),
    val = dict(
        data_prefix = 'data/val',
        ann_file = 'data/val.txt',
        classes = 'data/classes.txt'
    )
)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=100)

 # 预训练模型
load_from = 'checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'

