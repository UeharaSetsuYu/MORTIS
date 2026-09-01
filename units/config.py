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

    args = parser.parse_args()

    return args


def dataset_config(args):
    Dataset = args.dataset
    if Dataset == 'Caltech5V':
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

if __name__ == '__main__':
    args = parse_args()
    print(args)

