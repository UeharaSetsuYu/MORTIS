import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='mode of operation')
    parser.add_argument('--host', type=str, help='host address')
    parser.add_argument('--port', type=int, help='port number')


    parser.add_argument('--times', type=int, default=1 , help='Train times')
    parser.add_argument('--dataset', type=str, default='DHA', help='Dataset Name')
    parser.add_argument('--epochs', type=int, default='700', help='number of epochs')
    parser.add_argument('--batch_size', type=int, default='256', help='batch size')
    parser.add_argument('--train_rate', type=float, default=0.8, help='train data rate')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--lr', type=float, default=1.0e-4, help='learning rate')
    parser.add_argument('--pre_train', type=int, default=150, help='pre-train times')
    parser.add_argument('--model', type=str, default='Clustering', help='Or Classification')
    parser.add_argument('--missing_rate', type=float, default = 0.5, help='Incomplete data missing rate')
    parser.add_argument('--data_model', type=str, default='incomplete', help='incomplete or unpair')
    parser.add_argument('--anchor_view', type=int, default=0, help='anchor view for unpaired data')
    parser.add_argument('--test_times', type=int, default=100, help='Test times')
    # Hyper-parameters
    parser.add_argument('--lamda_1', type=float, default = 1, help='Hyper parameters')
    parser.add_argument('--lamda_2', type=float, default=1, help='Hyper parameters')
    parser.add_argument('--lamda_3', type=float, default=1, help='Hyper parameters')
    parser.add_argument('--alpha', type=float, default=0.1, help='Hyper parameters')
    parser.add_argument('--beta', type=float, default=0.1, help='Hyper parameters')
    parser.add_argument('--gamma', type=float, default=0.1, help='Hyper parameters')

    # Training stage
    parser.add_argument('--pre_train_epoch', type=int, default=150, help='pretraining epoch')
    parser.add_argument('--adversarial_epoch', type=float, default=200, help='adversarial-based consistency learning')


    # analysis
    parser.add_argument('--step_num', type = int, default = 5, help='number of steps')
    parser.add_argument('--top_k', type=int, default=5, help='number of steps')
    parser.add_argument('--warming_up', type = int, default = 150, help='warm up')
    parser.add_argument('--All_test', type = bool, default=False, help='test all models')


    # Dataset Noisy Setting
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'salt_pepper', 'dropout'])
    parser.add_argument('--noise_std', type=float, default=0.4) # GAUSSIAN STD IS  0.4
    parser.add_argument('--salt_pepper_ratio', type=float, default=0.1)
    parser.add_argument('--min_noisy_views', type=int, default=1)
    parser.add_argument('--max_noisy_views', type=int, default=1)

    ## PCC setting
    parser.add_argument('--PCC_Setting', type=str, default='PCC', help = "PCC or W-o-PCC")
    args = parser.parse_args()

    return args


def dataset_config(args):
    Dataset = args.dataset
    if Dataset == 'BDGP':
        return dict(
            view_num = 2,
            seed = 5,
            class_num = 5,
            epochs = 300,
            learning_rate = 1e-4,
            Autoencoder=dict(

                arch1=[1750, 1024, 1024, 1024, 128],
                arch2=[79, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 150,
            warming_up = 100,
        )
    elif Dataset == 'Caltech':  # Caltech101_20
        return dict(
            view_num = 6,
            seed = 5,
            class_num = 20,
            epochs = 400,
            learning_rate = 1e-4,
            Autoencoder=dict(
                arch1=[48, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                arch3=[254, 1024, 1024, 1024, 128],
                arch4=[1984, 1024, 1024, 1024, 128],
                arch5=[512, 1024, 1024, 1024, 128],
                arch6=[928, 1024, 1024, 1024, 128],

                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch=100,
            warming_up=100,
        )
    elif Dataset == 'BBCSport':
        return dict(
            view_num = 2,
            seed = 5,
            class_num = 5,
            epochs = 300,
            learning_rate = 1e-4,
            Autoencoder=dict(
                arch1=[3183, 1024, 1024, 1024, 128],
                arch2=[3203, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 544,
            pre_train_epoch = 200,
            warming_up=100,
        )
    elif Dataset == 'Scene_15':
        return dict(
            view_num = 3,
            seed = 5,
            class_num = 15,
            epochs = 300,
            learning_rate = 1e-4,
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                arch3=[40, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,

            ),
            batch_size = 1024,
            pre_train_epoch=100,
            warming_up=100,

        )
    elif Dataset == 'LandUse_21':
        return dict(
            view_num = 3,
            seed = 5,
            class_num = 21,
            epochs = 300,
            learning_rate = 1e-4,
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                arch3=[40, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 256,
            pre_train_epoch=100,
            warming_up=100,
        )
    elif Dataset == 'NGs':
        return dict(
            view_num=3,
            seed=5,
            class_num=5,
            epochs=800,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 128],
                arch2=[2000, 1024, 1024, 1024, 128],
                arch3=[2000, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 256
        )
    elif Dataset == 'DHA':
        return dict(
            view_num=2,
            seed=5,
            class_num=23,
            epochs=300,
            N = 483,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[110, 500, 500, 2000, 128],
                arch2=[6144, 500, 500, 2000, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 483,
            pre_train_epoch = 250,
            warming_up = 100,
            prototype_start_epoch = 20,
            prototype_warmup_epoch = 50,
            prototype_walk_steps = 0,   # Random self-walking step
            prototype_update_momentum = 0.9,
            unpaired_match_weight_scale = 0.3,
            mknn_intra_weight = 1.0,
            mknn_paired_cross_weight = 1.0,
            mknn_unpaired_cross_weight = 0.1,
            mknn_paired_topk = 3,
            mknn_unpaired_topk = 1,
            mknn_topk = 3,  # MkNN k, incomplete, noise and mix scenes.
            mknn_temperature = 0.2,
            mknn_unpaired_fallback = False,
        )
    elif Dataset == 'ALOI':
        return dict(
            view_num=4,
            seed=5,
            class_num=100,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[77, 1024, 1024, 1024, 128],
                arch2=[13, 1024, 1024, 1024, 128],
                arch3=[64, 1024, 1024, 1024, 128],
                arch4=[125, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 200,
            warming_up = 100,
        )
    elif Dataset == 'NUSWIDE':
        return dict(
            view_num=2,
            seed=5,
            class_num=6,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[4096, 1024, 1024, 1024, 128],
                arch2=[300, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 250,
            warming_up=100,

        )
    elif Dataset == 'CUB':
        return dict(
            view_num=2,
            seed=5,
            class_num=10,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[1024, 1024, 1024, 1024, 128],
                arch2=[300, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 600,
            pre_train_epoch = 100,
            warming_up=100,
            mknn_topk=3, # mknn config. true_K = 2 * mknn_topk
        )
    elif Dataset == 'Hdigit':
        return dict(
            view_num=2,
            seed=5,
            class_num=10,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[256, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,
            mknn_topk=3  # mknn config. true_K = 2 * mknn_topk

        )
    elif Dataset == 'cora':
        return dict(
            view_num=2,
            seed=5,
            class_num=7,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[2708, 1024, 1024, 1024, 128],
                arch2=[1433, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,

        )
    elif Dataset == 'Caltech101_7':
        return dict(
            view_num=6,
            seed=5,
            class_num=7,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[48, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                arch3=[254, 1024, 1024, 1024, 128],
                arch4=[1984, 1024, 1024, 1024, 128],
                arch5=[512, 1024, 1024, 1024, 128],
                arch6=[928, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 512,
            pre_train_epoch = 100,
            warming_up=100,

        )
    elif Dataset == 'Caltech5V':
        return dict(
            view_num=5,
            seed=5,
            class_num=7,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[40, 1024, 1024, 1024, 128],
                arch2=[254, 1024, 1024, 1024, 128],
                arch3=[1984, 1024, 1024, 1024, 128],
                arch4=[512, 1024, 1024, 1024, 128],
                arch5=[928, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,
            mknn_topk=3,  # mknn config. true_K = 2 * mknn_topk
        )
    elif Dataset == 'HW':
        return dict(
            view_num=6,
            seed=5,
            class_num=10,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[216, 1024, 1024, 1024, 128],
                arch2=[76, 1024, 1024, 1024, 128],
                arch3=[64, 1024, 1024, 1024, 128],
                arch4=[6, 1024, 1024, 1024, 128],
                arch5=[240, 1024, 1024, 1024, 128],
                arch6=[47, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 512,
            pre_train_epoch = 100,
            warming_up=100,
        )
    elif Dataset == 'Reuters_small':
        return dict(
            view_num=5,
            seed=5,
            class_num=6,
            epochs=200,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 128],
                arch2=[2000, 1024, 1024, 1024, 128],
                arch3=[2000, 1024, 1024, 1024, 128],
                arch4=[2000, 1024, 1024, 1024, 128],
                arch5=[2000, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 512,
            pre_train_epoch = 100,
            warming_up=100,
        )
    elif Dataset == 'NUS_WIDE':
        return dict(
            view_num=5,
            seed=5,
            class_num=31,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[65, 1024, 1024, 1024, 128],
                arch2=[226, 1024, 1024, 1024, 128],
                arch3=[145, 1024, 1024, 1024, 128],
                arch4=[74, 1024, 1024, 1024, 128],
                arch5=[129, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,
        )
    elif Dataset == 'MSRC_v1':
        return dict(
            view_num=5,
            seed=5,
            class_num=7,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[24, 1024, 1024, 1024, 128],
                arch2=[576, 1024, 1024, 1024, 128],
                arch3=[512, 1024, 1024, 1024, 128],
                arch4=[256, 1024, 1024, 1024, 128],
                arch5=[254, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 210,
            pre_train_epoch = 100,
            warming_up=100,
        )
    elif Dataset == 'flower17':
        return dict(
            view_num=7,
            seed=5,
            class_num=17,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[1360, 1024, 1024, 1024, 128],
                arch2=[1360, 1024, 1024, 1024, 128],
                arch3=[1360, 1024, 1024, 1024, 128],
                arch4=[1360, 1024, 1024, 1024, 128],
                arch5=[1360, 1024, 1024, 1024, 128],
                arch6=[1360, 1024, 1024, 1024, 128],
                arch7=[1360, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,
        )

    elif Dataset == 'NoisyMNIST':
        return dict(
            view_num=2,
            seed=5,
            class_num=10,
            epochs=300,
            learning_rate=1e-4,
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations='relu',
                batchnorm=True,
            ),
            batch_size = 1024,
            pre_train_epoch = 100,
            warming_up=100,
        )
if __name__ == '__main__':
    args = parse_args()
    print(args)

