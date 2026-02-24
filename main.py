import argparse
import os
import torch
import math
from exp.exp_main import Exp_Main

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.metrics import metric

def set_seed(seed):
    random.seed(seed)
    seed += 1
    np.random.seed(seed)
    seed += 1
    torch.manual_seed(seed)

battery_EOL = {
    'B0005': [125],
    'B0006': [109],
    'B0007': [144],
    'B0018': [97],
}

for K in [1,2,3,4,5]:
    battery = 'B0006'
    battery_name = battery
    per_cent = 0.2
    seq_len = 50
    class Args:
        def __init__(self):
            self.model_id = 'xx'
            self.name = '{}K_'.format(K)
            self.model = 'FF'
            self.method = '1'
            self.data = 'NASA'
            self.end = 1.4
            self.factor = K

            self.battery_EOL = battery_EOL
            self.per_cent = per_cent
            self.battery_name = battery_name
            self.root_path = './datasets/NASA'
            self.results_path = './results/K_compare/'
            self.data_path = 'xx'
            self.battery_path = 'battery_data_frames[{}].csv'.format(battery_name)
            self.checkpoints = 'E:/Python_codes/RUL/RUL/ETS_checkpoints/'

            # self.seq_len = math.ceil(battery_EOL[battery_name][0] * per_cent)
            self.seq_len = seq_len
            self.batch_size = 32
            self.patience = 15
            self.epochs = 30
            self.learning_rate = 5e-4
            self.d_model = 2048
            self.dropout = 0.1
            self.d_ff = 2048
            self.n_heads = 1
            self.e_layers = 2
            self.d_layers = 2
            self.warmup_epochs = 10
            self.pred_len = 1


            self.label_len = 10
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.std = 0.2
            self.activation = 'sigmoid'
            self.min_lr = 2e-30
            self.smoothing_learning_rate = 0
            self.damping_learning_rate = 0
            self.output_attention = False
            self.optim = 'adam'
            self.itr = 1
            self.des = 'test'
            self.lradj = 'exponential_with_warmup'
            self.use_gpu = True
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0,1,2,3'
            self.all_train_time = 0
            self.all_test_time = 0

    args = Args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Main

    datasets = ['svmd2_{}.csv'.format(battery), 'svmd3_{}.csv'.format(battery), 'svmd1_{}.csv'.format(battery)]
    # datasets = [ 'svmd1_{}.csv'.format(battery)]
    if args.method == 'no':
        datasets = ['battery_data_frames[{}].csv'.format(battery_name)]
    all_results = {}

    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"Starting training and testing for dataset: {dataset}")
        print(f"{'=' * 50}\n")

        args.data_path = dataset

        set_seed(42)

        exp = Exp(args)

        print(f'<<<<<<<<<<<<<<<<<<<<<<<<< Training on {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>')
        results = exp.train()

        # print(f'<<<<<<<<<<<<<<<<<<<<<<<<< Testing on {dataset} >>>>>>>>>>>>>>>>>>>>>>>>>')
        # results = exp.test(Time_record=False)

        all_results[dataset] = results

        torch.cuda.empty_cache()

    print("\nFinal Results Summary:")
    final_results = 0
    for dataset, result in all_results.items():
        print(f"\nResults for {dataset}:")
        final_results = final_results + result


    def save_predictions_to_excel(folder_path, short_name, preds, trues):
        # Determine the maximum length between preds and trues
        max_len = len(trues)

        # Pad the shorter array with NaN values to make them the same length
        padding_length = max_len - len(preds)
        padded_preds = np.concatenate([[np.nan] * padding_length, preds])

        # Create a DataFrame
        df = pd.DataFrame({
            'True Values': trues,
            'Predicted Values': padded_preds
        })

        # Save the DataFrame to an Excel file
        excel_path = os.path.join(folder_path, f'{short_name}.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Predictions and true values saved to {excel_path}")

    def draw_SOH(pred_values, start_idx, folder_path, file_name):

        preds_plot = pred_values
        root_path = './datasets/'
        df_raw = pd.read_csv(os.path.join(root_path, args.battery_path))
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        total_true_values = df_data['capacity'].values
        length = len(pred_values)
        total_true = total_true_values[start_idx:][:length]

        mae, mse, rmse, mape, mspe = metric(preds_plot, total_true)
        preds_index = np.argmax(pred_values < args.end) + 1
        trues_index = args.battery_EOL[battery_name][0] - args.seq_len
        ae = preds_index - trues_index
        re = abs(ae / trues_index) * 100
        loss_file_path = os.path.join(folder_path, 'loss_{}.txt'.format(battery_name))
        with open(loss_file_path, 'w') as file:
            file.write('loss:' + '\n')
            file.write('MAE:' + str(mae) + '\n')
            file.write('MSE:' + str(mse) + '\n')
            file.write('RMSE:' + str(rmse) + '\n')
            file.write('MAPE:' + str(mape) + '\n')
            file.write('AE:' + str(ae) + '\n')
            file.write('RE: {:.3f}%\n'.format(re))
        print('MAE:{:.7f}, MSE:{:.7f}, REMSE:{:.7f}'.format(mae, mse, rmse))
        print('rea:{}, pre:{}, AE:{}, RE:{:.3f}%'.format(trues_index, preds_index, ae, re))
        plt.figure(figsize=(15, 5))

        plt.plot(total_true_values, label='Total True Values')

        if args.data == 'NASA':
            plt.plot(range(start_idx, start_idx + len(preds_plot)), preds_plot, label='Predicted (Test Part)',
                     color='red', linestyle='--', linewidth=2)

        if args.data == 'CALCE':
            plt.plot(range(start_idx, start_idx + len(preds_plot)), preds_plot, label='Predicted (Test Part)',
                     color='red')

        plt.axhline(y=args.end, color='blue', linestyle='--', label='Value = {}'.format(args.end))

        plt.legend()
        plt.title('RMSE_loss:{:.5f}'.format(rmse))
        plt.xlabel('Cycle')
        plt.ylabel('SOH(%)')
        plt.grid(True)
        plt.savefig(folder_path +'/' + file_name +'.png')
        plt.show()
        save_predictions_to_excel(folder_path, args.battery_name, pred_values, total_true_values)


    folder_path = args.results_path + args.model_id + '/'
    draw_SOH(final_results, args.seq_len, folder_path, args.battery_name)





