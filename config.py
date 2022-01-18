import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str, default='./weight/erasing_net', help='models weight are saved here')
    parser.add_argument('--log_root', type=str, default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--s_model', type=str, default='./weight/s_net/WRN-16-1-S-model_best.pth.tar', help='path of student model')
    parser.add_argument('--t_model', type=str, default='./weight/t_net/WRN-16-1-T-model_best.pth.tar', help='path of teacher model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05, help='ratio of training data')
    parser.add_argument('--beta1', type=int, default=500, help='beta of low layer')
    parser.add_argument('--beta2', type=int, default=1000, help='beta of middle layer')
    parser.add_argument('--beta3', type=int, default=1000, help='beta of high layer')
    parser.add_argument('--p', type=float, default=2.0, help='power for AT')
    parser.add_argument('--threshold_clean', type=float, default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float, default=90.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str, default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str, default='WRN-16-1', help='name of teacher')
    parser.add_argument('--s_name', type=str, default='WRN-16-1', help='name of student')

    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=5, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    return parser
