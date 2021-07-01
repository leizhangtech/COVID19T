import argparse


def parse_arguments():
    """Argument Parser for the commandline argments
    :returns: command line arguments

    """
    ##########################################################################
    #                            Training setting                            #
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', help="Number of classes", type=int,
                        default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float,
                        help='LR Multiplicative factor if lr_scheduler is step',
                        default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--backbone', type=str, default='Effv2', choices=['Than', 'COV', 'BiT', 'Effv2'])
    args = parser.parse_args()

    return args
