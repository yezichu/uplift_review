import numpy as np
from pathlib import Path
from torch.utils.data import random_split
import argparse  
from model.S_learner import SLearnerEstimator
from model.T_learner import TLearnerEstimator
from model.X_learner import XLearnerEstimator
from data.dataset import load_ihdp_data, IPDHDataset, IPHDDataLoader
from utils.metric import pehe, policy_risk, uplift_curve, uplift_auc_score, qini_curve_industry, qini_auc_score_industry
from utils.plot import plot_uplift_curve, plot_qini_curve


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", default = "x-learner", type = str)
    args = parse.parse_args()
    
    train_path = Path('data/ihdp_npci_1-100.train.npz')
    data_dict = load_ihdp_data(train_path)
    full_dataset = IPDHDataset(data_dict)
    train_ratio = 0.8
    train_size = int(train_ratio * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_set, valid_set = random_split(full_dataset, [train_size, valid_size])
    train_loader = IPHDDataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = IPHDDataLoader(valid_set, batch_size=32, shuffle=False)

    test_path = Path('data/ihdp_npci_1-100.test.npz')
    test_data = load_ihdp_data(test_path)
    test_set = IPDHDataset(test_data)
    test_loader = IPHDDataLoader(test_set, batch_size=32, shuffle=False)
    
    if args.model == 's-learner':
        slearner = SLearnerEstimator(x_dim = 25, hidden_dim = 64, lr=1e-3)
        slearner.fit(train_loader, valid_loader, 100)
        tau_hat_list, tau_true_list = slearner.evaluate(test_loader)
    elif args.model == 't-learner':
        tlearner = TLearnerEstimator(x_dim = 25, hidden_dim = 64, lr=1e-3)
        tlearner.fit(train_loader, valid_loader, 100)
        tau_hat_list, tau_true_list = tlearner.evaluate(test_loader)   
    elif args.model == 'x-learner':
        xlearner = XLearnerEstimator(x_dim = 25, hidden_dim = 64)
        xlearner.fit(train_loader, valid_loader, 100, lr=1e-3)
        tau_hat_list, tau_true_list = xlearner.evaluate(test_loader)
        
    pehe_score = pehe(tau_hat_list, tau_true_list)
    print("PEHE:", pehe_score) 
    
    
if __name__ == "__main__":
    main()
