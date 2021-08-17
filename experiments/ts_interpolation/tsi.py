import os, math
import torch
from dataclasses import asdict

from models.ts_ae.training import construct_tsae, train, test
from utils.exp import construct_logger_and_dir, save_yaml_file
from utils.ts_interpolation import TSIExpLog, construct_dataset_and_loaders


def tsi_experiment(args):

    # construct logger, exp directory, and set device
    torch.set_printoptions(precision = 3, sci_mode = False)
    log_file_name = 'tsi'
    logger, full_log_dir = construct_logger_and_dir(log_file_name, args.log_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # LP: cuda or AV: cuda:0
    args.log_dir = full_log_dir
    args.log_file_name = log_file_name + '.log'
    args.device = device

    # save experiment params in yaml file
    yaml_file_name = os.path.join(full_log_dir, 'experiment_parameters.yml')
    save_yaml_file(yaml_file_name, asdict(args))

    # data and model
    dataset, train_loader, test_loader = construct_dataset_and_loaders(args)
    tsae, tsae_opt, loss_func = construct_tsae(dataset, args, device)

    # training
    best_test_loss = math.inf
    best_epoch = 0
    exp_log = TSIExpLog(args)

    logger.info("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train(tsae, train_loader, loss_func, tsae_opt)
        logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

        if epoch % args.test_interval == 0:
            test_loss, test_log = test(tsae, test_loader, loss_func)
            logger.info("Epoch {}: Test Loss {}".format(epoch, test_loss))       
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(tsae.state_dict(), os.path.join(full_log_dir, 'best_model.pt'))
            exp_log.add_test_log(test_log)

    logger.info("Best Test Loss {} on Epoch {}".format(best_test_loss, best_epoch))
    exp_log.completed = True

    return exp_log

### placeholders while I figure out the display functions

def display_train_and_test_embs(train_log, test_log, which_to_print):
    # which_to_print is dict
    fig, axs = plt.subplots(1, 2, figsize = (16, 8))
    display_embs(train_log, fig, axs[0], which_to_print, 'Train')
    display_embs(test_log, fig, axs[1], which_to_print, 'Test')
    return fig, axs

def display_embs(exp_log, fig, ax, which_to_print, lab):
    # TODO? fig 4: compare test vs train embeddings, plot cosine loss
    # ae_log_dict = {r/p/ts_gt/ts_premap/ts_postmap : (mapped, decoded); batch_node_vecs : batch_node_vecs}
    # mapped = node_emb, edge_emb, graph_emb, coord_out
    # decoded = recon_node_fs, recon_edge_fs, adj_pred

    r, p, ts_gt, ts_premap, ts_postmap = which_to_print['r'], which_to_print['p'], \
        which_to_print['ts_gt'], which_to_print['ts_premap'], which_to_print['ts_postmap']
    final_res_batched = exp_log.epoch_ae_results[-1] # = [{batch_res}, {batch_res}, .., {batch_res}]
    graph_embs = {'r': [], 'p': [], 'ts_gt': [], 'ts_premap': [], 'ts_postmap': [], 'ts_node': []}
    
    for batch_res in final_res_batched:
        node_embs = batch_res['ts_postmap'][0][0]
        ts_batch_vec = batch_res['batch_node_vecs'][2]
        ts_node_emb_batch = to_dense_batch(node_embs, ts_batch_vec)[0] # [0] cos just append tensors, not true/false values

        # graph embs
        r_graph_embs, p_graph_embs = batch_res['r'][0][2], batch_res['p'][0][2]
        if ts_gt:
            ts_gt_graph_embs = batch_res['ts_gt'][0][2]
        if ts_premap:
            ts_premap_graph_embs = batch_res['ts_premap'][0][2]
        if ts_postmap:
            ts_postmap_graph_embs = batch_res['ts_postmap'][0][2]

        for mol_id, ts_node_emb in enumerate(ts_node_emb_batch):
            graph_embs['ts_node'].append(ts_node_emb)
            graph_embs['r'].append(r_graph_embs[mol_id].detach().numpy())
            graph_embs['p'].append(p_graph_embs[mol_id].detach().numpy())
            if ts_gt:
                graph_embs['ts_gt'].append(ts_gt_graph_embs[mol_id].detach().numpy())
            if ts_premap:
                graph_embs['ts_premap'].append(ts_premap_graph_embs[mol_id].detach().numpy())
            if ts_postmap:
                graph_embs['ts_postmap'].append(ts_postmap_graph_embs[mol_id].detach().numpy())

    # fig, ax = plt.subplots(figsize = (8, 8))

    # colours, scatter plot
    cols = {'r': 'red', 'p': 'green', 'ts_gt': 'orange', 'ts_premap': 'yellow', 'ts_postmap': 'blue'}
    if r:
        ax.scatter(*zip(*graph_embs['r']), color = cols['r'])
    if p:
        ax.scatter(*zip(*graph_embs['p']), color = cols['p'])
    if ts_gt:
        ax.scatter(*zip(*graph_embs['ts_gt']), color = cols['ts_gt'])
    if ts_premap:
        ax.scatter(*zip(*graph_embs['ts_premap']), color = cols['ts_premap'])
    if ts_postmap:
        ax.scatter(*zip(*graph_embs['ts_postmap']), color = cols['ts_postmap'])
    markers = [plt.Line2D([0,0], [0,0], color = color, marker = 'o', linestyle = '') for color in  cols.values()]
    ax.legend(markers, cols.keys())

    # title, axes
    num_rxns = exp_log.num_rxns
    train_ratio = exp_log.tt_split
    batch_size = exp_log.batch_size
    epochs = exp_log.get_performed_epochs()
    title = f"[{lab}] {num_rxns} Reactions, Train Ratio: {train_ratio}, {epochs} Epochs, Batch Size: {batch_size}"
    ax.set_title(title)
    ax.set_ylabel('Graph Emb Dim 1')
    ax.set_xlabel('Graph Emb Dim 2')

    return fig, ax






