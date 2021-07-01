import os
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

import model
import config
import dataset
import utils
from logger import create_loggerL
import torch.distributed as dist

# call the evaluate_model to get the predictions
def evaluate_model(model, val_loader, logger):
    metric = torch.nn.CrossEntropyLoss()
    model.eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    y_probs = np.zeros((0, args.n_classes), float)
    losses, y_trues = [], []

    for i, (image, label, case_id) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1).detach().cpu().numpy()

        y_probs = np.concatenate([y_probs, y_prob])
        y_trues.append(label.item())
        logger.info(f'{case_id[0][:-2]}, {y_prob[0]}, { np.argmax(y_prob, axis=1)[0]}')
    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    val_loss_epoch = np.mean(losses)
    return val_loss_epoch, metric_collects




def main(args):
    """Main function for the testing pipeline

    :args: commandline arguments
    :returns: None

    """
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experimentsThanT'
    model_dir = os.path.join(exp_dir, 'models')
    result_dir = os.path.join(exp_dir, 'result') # the folder to save the prediction results
    test_result = os.path.join(result_dir, 'test_result.txt')   #save the predictions to the test_result file
    model_file = os.path.join(model_dir, 'best.pth')
    os.makedirs(result_dir, exist_ok=True)
    logger = create_loggerL(output_dir=result_dir, dist_rank=0, name=f"{test_result}")

    val_dataset = dataset.NCovDataset('../../dataset/MIA-COV19-DATA/Data/Lung/ICCV_Lung_split_test/data/', stage='val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=11,
        drop_last=False)

    if exp_dir =='experimentsThanT':
        cov_net = model.COVNetT(n_classes=args.n_classes)
    elif exp_dir == 'experimentsBiT':
        cov_net = model.COVNetBiT(n_classes=args.n_classes)
    elif exp_dir == 'experimentsEffv2':
        cov_net = model.COVNetEffi(n_classes=args.n_classes)
    if torch.cuda.is_available():
        cov_net.cuda()

    state = torch.load(model_file)

    cov_net.load_state_dict(state.state_dict())

    with torch.no_grad():
        val_loss, metric_collects = evaluate_model(cov_net, val_loader, logger)
    prefix = '******Evaluate******'
    utils.print_progress(mean_loss=val_loss, metric_collects=metric_collects,
                         prefix=prefix)



if __name__ == "__main__":
    args = config.parse_arguments()
    main(args)
