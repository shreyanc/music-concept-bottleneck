import pandas as pd
import torch
import numpy as np
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F
import logging
from itertools import islice
import torch.nn as nn

from helpers.loss_funcs import VarCriterion, ScaledCriterion
from utils import compute_metrics
import composer.functional as cf

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, epoch, run_name, **kwargs):
    model.train()
    loss_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=len(dataloader), desc=run_name):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        output = model(inputs)
        try:
            loss = criterion(output['output'].float(), labels.float())
        except Exception as e:
            print(e)
            loss = criterion(output.float(), labels.float())

        if kwargs.get('loss_batch_average'):
            loss = torch.mean(torch.sum(loss, axis=-1))

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(loss_list)

    return {'avg_loss': epoch_loss}


def train_explainable(model, dataloader_non_concept, dataloader_concept, optimizer_non_concept, optimizer_concept, criterion, k, epoch, run_name, **kwargs):
    model.train()
    loss_list_main = []
    loss_list_concepts = []

    concept_iterator = iter(dataloader_concept)

    for batch_idx, batch in tqdm(enumerate(dataloader_non_concept), ascii=False, total=len(dataloader_non_concept), desc=run_name):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer_non_concept.zero_grad()
        output = model(inputs)
        try:
            loss = criterion(output['output'].float(), labels.float())
        except Exception as e:
            print(e)
            loss = criterion(output.float(), labels.float())

        loss_list_main.append(loss.item())
        loss.backward()
        optimizer_non_concept.step()

        if batch_idx % k == k - 1:
            # print("TRAINING MIDLEVEL LAYER")
            song_ids, inputs, labels = next(concept_iterator)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            optimizer_concept.zero_grad()
            output = model(inputs)
            try:
                loss = criterion(output['concepts'].float(), labels.float())
            except Exception as e:
                print(e)
                loss = criterion(output.float(), labels.float())
            loss_list_concepts.append(loss.item())
            loss.backward()
            optimizer_concept.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts}


def train_explainable_mixed_batches(model, dataloader, optimizer, tasks_list, criteria, run_name, epoch, **kwargs):
    model.train()

    loss_list = []
    loss_list_main = []
    loss_list_concepts = []
    dlen = kwargs['dataloader_len']

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=dlen, desc=run_name):
        samples = []
        for x in batch:
            samples.append((x[1].to(device), x[2].to(device)))

        inputs = torch.cat([s[0] for s in samples])
        labels = [s[1] for s in samples]
        sizes = [s[0].shape[0] for s in samples]

        optimizer.zero_grad()
        output = model(inputs.unsqueeze(1))
        main_preds = output['output'][:sizes[0]]
        main_labels = labels.pop(0)
        main_loss = criteria[0](main_preds.float(), main_labels.float())

        all_concept_preds = iter(output['concepts'][sizes[0]:])
        concept_preds = [list(islice(all_concept_preds, length)) for length in sizes[1:]]
        concept_dims = [i.shape[1] for i in labels]
        concept_idxs = [0, concept_dims[0]] + [concept_dims[i] + concept_dims[i - 1] for i in range(1, len(concept_dims))]

        concept_losses = []
        for cidx, closs in enumerate(criteria[1:]):
            cur_concept_preds = torch.stack(concept_preds[cidx])[:, concept_idxs[cidx]: concept_idxs[cidx + 1]]
            cur_labels = labels[cidx]
            cur_loss = closs(cur_concept_preds.float(), cur_labels.float())
            if kwargs.get('loss_batch_average'):
                cur_loss = torch.mean(torch.sum(cur_loss, axis=-1))
            # concept_loss += cur_loss
            concept_losses.append(cur_loss)

        # loss = (1 - alpha) * main_loss + alpha * concept_loss
        # loss = main_loss + sum(concept_losses)

        all_losses = [main_loss] + concept_losses
        log_dict = {f'loss_{tasks_list[i]}': all_losses[i] for i in range(len(all_losses))}
        log_dict.update({"Iteration": batch_idx + (epoch - 1) * dlen})
        wandb.log(log_dict)

        loss = sum(all_losses)

        loss_list_main.append(main_loss.item())
        # loss_list_concepts.append(concept_loss.item())
        loss_list_concepts.append(sum(concept_losses).item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts}


def train_explainable_mixed_batches_var_loss(model, dataloader, optimizer, tasks_list, criteria, run_name, epoch, **kwargs):
    model.train()
    loss_list = []
    loss_list_main = []
    loss_list_concepts = []
    dlen = kwargs['dataloader_len']
    num_labels_dict = {'ml': 7, 'ins': 41, 'mood': 56, 'genre': 95, 'deam': 2, 'deam+pmemo': 2}

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=dlen, desc=run_name):
        samples = []
        for task_xy in batch:
            # len(batch) == number of tasks
            # task_xy[0] == filenames
            # task_xy[1] == input spectrogram
            # task_xy[2] == labels
            samples.append((task_xy[1].to(device), task_xy[2].to(device)))

        inputs = torch.cat([s[0] for s in samples])
        labels = [s[1] for s in samples]
        sizes = [s[0].shape[0] for s in samples]
        out_indices = np.hstack([[0], np.cumsum(sizes)])
        concept_dims = [i.shape[1] for i in labels[1:]]
        concept_dims_indices = np.hstack([[0], np.cumsum(concept_dims)])

        optimizer.zero_grad()
        output = model(inputs.unsqueeze(1))
        task_preds = []
        for taskidx, taskname in enumerate(tasks_list):
            task_output_len = num_labels_dict[taskname]
            if taskidx == 0:  # main task
                cur_task_preds = output['output'][out_indices[taskidx]: out_indices[taskidx + 1]]
            else:  # concepts
                cur_task_preds = output['concepts'][out_indices[taskidx]: out_indices[taskidx + 1]]
                cur_task_preds = cur_task_preds[:, concept_dims_indices[taskidx - 1]: concept_dims_indices[taskidx]]
            assert task_output_len == cur_task_preds.shape[1]
            task_preds.append(cur_task_preds)

        combined_loss, all_losses, all_scaled_losses = criteria(task_preds, labels)

        log_dict = {f'loss_{tasks_list[i]}': all_losses[i] for i in range(len(all_losses))}
        log_dict.update({f'scaled_loss_{tasks_list[i]}': all_scaled_losses[i] for i in range(len(all_scaled_losses))})
        if isinstance(criteria, VarCriterion):
            log_dict.update({f'logvar_{tasks_list[i]}': criteria.log_vars[i] for i in range(len(criteria.log_vars))})

        log_dict.update({"Iteration": batch_idx + (epoch - 1) * dlen})
        wandb.log(log_dict)

        loss_list.append(combined_loss.item())
        loss_list_main.append(all_losses[0].detach().cpu().numpy())
        loss_list_concepts.append(torch.sum(torch.stack(all_losses[1:])).detach().cpu().numpy())
        combined_loss.backward()
        optimizer.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts}


def train_explainable_mixed_batches_regularised(model, dataloader, optimizer, criteria, tasks_list, run_name, reg_loss, reg_weight=1.0, **kwargs):
    model.train()

    if (alpha := kwargs.get('alpha')) is None:
        alpha = 0.5  # alpha: concept loss weight
    else:
        if not 0.0 <= alpha <= 1.0:
            logger.warning(f"Alpha is out of range [0,1] alpha={alpha}; defaulting to alpha=0.5")

    loss_list = []
    loss_list_main = []
    loss_list_concepts = []
    loss_list_extranode = []
    dlen = kwargs['dataloader_len']

    if reg_loss == 'l2':
        reg_loss_func = nn.MSELoss().to(device)
    elif reg_loss == 'l1':
        reg_loss_func = nn.L1Loss().to(device)
    else:
        logger.warning("Extranode weight regularisation loss not specified, defaulting to L1 Loss")
        reg_loss_func = nn.L1Loss().to(device)

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=dlen, desc=run_name):
        samples = []
        for x in batch:
            samples.append((x[1].to(device), x[2].to(device)))

        inputs = torch.cat([s[0] for s in samples])
        labels = [s[1] for s in samples]
        sizes = [s[0].shape[0] for s in samples]

        optimizer.zero_grad()
        output = model(inputs.unsqueeze(1))
        main_preds = output['output'][:sizes[0]]
        main_labels = labels.pop(0)
        main_loss = criteria[0](main_preds.float(), main_labels.float())

        if kwargs.get('loss_batch_average'):
            main_loss = torch.mean(torch.sum(main_loss, axis=-1))

        all_concept_preds = iter(output['concepts'][sizes[0]:])
        concept_preds = [list(islice(all_concept_preds, length)) for length in sizes[1:]]
        concept_dims = [i.shape[1] for i in labels]
        concept_idxs = [0, concept_dims[0]] + [concept_dims[i] + concept_dims[i - 1] for i in range(1, len(concept_dims))]

        concept_loss = 0
        for cidx, closs in enumerate(criteria[1:]):
            cur_concept_preds = torch.stack(concept_preds[cidx])[:, concept_idxs[cidx]: concept_idxs[cidx + 1]]
            cur_labels = labels[cidx]
            cur_loss = closs(cur_concept_preds.float(), cur_labels.float())
            if kwargs.get('loss_batch_average'):
                cur_loss = torch.mean(torch.sum(cur_loss, axis=-1))
            concept_loss += cur_loss

        loss = (1 - alpha) * main_loss + alpha * concept_loss

        num_extra_nodes = output['concepts'][sizes[0]:].shape[-1] - concept_idxs[-1]
        if num_extra_nodes == 1:
            w = model.feed_forward.weight[:, -1]
            reg_loss = reg_weight * reg_loss_func(w, torch.zeros(w.size()).to(device))
            loss_list_extranode.append(reg_loss.item())
            loss += reg_loss

        loss_list_main.append(main_loss.item())
        loss_list_concepts.append(concept_loss.item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)
    epoch_loss_extranode = np.mean(loss_list_extranode)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts, 'avg_loss_extranode': epoch_loss_extranode}


def train_explainable_mixed_batches_dwa(model, dataloader, optimizer, criteria, epoch, avg_cost_ep, run_name, **kwargs):
    model.train()

    loss_list = []
    loss_list_main = []
    loss_list_concepts = []
    all_losses_list = []
    dlen = kwargs['dataloader_len']

    num_tasks = len(criteria)

    if (T := kwargs.get('temp')) is None:
        T = 2.0

    if epoch == 1 or epoch == 2:  # Epochs are 1-indexed
        lambda_weight = [1.0] * num_tasks
    else:
        ws = [avg_cost_ep[epoch - 2][t] / avg_cost_ep[epoch - 3][t] for t in range(num_tasks)]  # Epochs are 1-indexed
        tot_w = sum([np.exp(ws[t] / T) for t in range(num_tasks)])
        lambda_weight = [num_tasks * np.exp(ws[t] / T) / tot_w for t in range(num_tasks)]

    logger.info(f"Lambdas for epoch {epoch}: {lambda_weight}")
    wandb.log({'main_lambda': lambda_weight[0], 'concept_lambda': lambda_weight[1]})

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=dlen, desc=run_name):
        samples = []
        for x in batch:
            samples.append((x[1].to(device), x[2].to(device)))

        inputs = torch.cat([s[0] for s in samples])
        labels = [s[1] for s in samples]
        sizes = [s[0].shape[0] for s in samples]

        optimizer.zero_grad()
        output = model(inputs.unsqueeze(1))
        main_preds = output['output'][:sizes[0]]
        main_labels = labels.pop(0)
        main_loss = criteria[0](main_preds.float(), main_labels.float())

        all_concept_preds = iter(output['concepts'][sizes[0]:])
        concept_preds = [list(islice(all_concept_preds, length)) for length in sizes[1:]]
        concept_dims = [i.shape[1] for i in labels]
        concept_idxs = [0, concept_dims[0]] + [concept_dims[i] + concept_dims[i - 1] for i in range(1, len(concept_dims))]

        concept_losses = []
        for cidx, closs in enumerate(criteria[1:]):
            cur_concept_preds = torch.stack(concept_preds[cidx])[:, concept_idxs[cidx]: concept_idxs[cidx + 1]]
            cur_labels = labels[cidx]
            concept_losses.append(closs(cur_concept_preds.float(), cur_labels.float()))

        loss = lambda_weight[0] * main_loss + sum([lambda_weight[i] * concept_losses[i - 1] for i in range(1, num_tasks)])

        loss_list_main.append(lambda_weight[0] * main_loss.item())
        loss_list_concepts.append(sum([lambda_weight[i] * concept_losses[i - 1].item() for i in range(1, num_tasks)]))
        loss_list.append(loss.item())
        all_losses_list.append([main_loss.item(), *[l.item() for l in concept_losses]])
        loss.backward()
        optimizer.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)
    epoch_all_losses = np.mean(np.vstack(loss_list_concepts), axis=1)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts, 'avg_all_losses': epoch_all_losses}


def train_explainable_mixed_batches_da(model, dataloader, optimizer, main_criterion, concept_criterion, epoch, run_name, **kwargs):
    model.train()
    loss_list = []
    loss_list_main = []
    loss_list_concepts = []
    dlen = kwargs['dataloader_len']

    p = epoch / 20
    lambda_ = 2. / (1 + np.exp(-10 * p)) - 1

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=dlen, desc=run_name):
        (_, inputs_main, labels_main), (_, inputs_concept, labels_concept), (_, inputs_target) = batch
        inputs_main, labels_main, inputs_concept, labels_concept, inputs_target = inputs_main.to(device), labels_main.to(device), inputs_concept.to(
            device), labels_concept.to(
            device), inputs_target.to(device)

        inputs = torch.cat([inputs_main, inputs_concept, inputs_target])
        inputs = inputs.unsqueeze(1)

        domain_y = torch.cat([torch.ones(inputs_main.shape[0] + inputs_concept.shape[0]),
                              torch.zeros(inputs_target.shape[0])])
        domain_y = domain_y.to(device)

        num_inputs_main = inputs_main.shape[0]
        num_inputs_concept = inputs_concept.shape[0]
        num_inputs_domain = inputs_target.shape[0]

        optimizer.zero_grad()
        output = model(inputs, num_labeled=num_inputs_main, lambda_=lambda_)
        main_preds = output['output'][:num_inputs_main]
        concept_preds = output['concepts'][num_inputs_main:num_inputs_main + num_inputs_concept]
        domain_preds = output['domain']

        main_loss = main_criterion(main_preds.float(), labels_main.float())
        concept_loss = concept_criterion(concept_preds.float(), labels_concept.float())
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds.squeeze(), domain_y)
        loss = main_loss + concept_loss + domain_loss

        loss_list_main.append(main_loss.item())
        loss_list_concepts.append(concept_loss.item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss_main = np.mean(loss_list_main)
    epoch_loss_concepts = np.mean(loss_list_concepts)

    return {'avg_loss_main': epoch_loss_main, 'avg_loss_concepts': epoch_loss_concepts}


def train_multihead(model, dataloader, optimizer, criterion, epoch, run_name, **kwargs):
    model.train()
    loss_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=len(dataloader), desc=run_name):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        output = model(inputs)
        try:
            loss = criterion(output['output']['pmemo'].float(), labels.float())
        except Exception as e:
            print(e)
            loss = criterion(output.float(), labels.float())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(loss_list)

    return {'avg_loss': epoch_loss}


def train_da_backprop(model, dataloader, optimizer, criterion, epoch, writer, run_name, **kwargs):
    model.train()
    loss_list = []
    ml_loss_list = []
    domain_loss_list = []

    dlen = kwargs['dataloader_len']

    p = epoch / 20
    lambda_ = 2. / (1 + np.exp(-10 * p)) - 1
    # lambda_ *= 0.1

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=dlen, desc=run_name):
        (song_ids_ml, inputs_ml, labels), (song_names_p, inputs_piano) = batch
        inputs_ml, labels, inputs_piano = inputs_ml.to(device), labels.to(device), inputs_piano.to(device)
        inputs = torch.cat([inputs_ml, inputs_piano])
        inputs = inputs.unsqueeze(1)

        domain_y = torch.cat([torch.ones(inputs_ml.shape[0]),
                              torch.zeros(inputs_piano.shape[0])])
        domain_y = domain_y.to(device)

        optimizer.zero_grad()
        output = model(inputs, num_labeled=inputs_ml.shape[0], lambda_=lambda_)
        label_preds = output['output']
        domain_preds = output['domain']
        # emb = output['embedding']

        ml_loss = criterion(label_preds.float(), labels.float())
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds.squeeze(), domain_y)
        loss = ml_loss + domain_loss

        ml_loss_list.append(ml_loss.item())
        domain_loss_list.append(domain_loss.item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    writer.add_scalar(f"{kwargs.get('phase', 'training')}/lambda", lambda_, epoch)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/ml_loss", np.mean(ml_loss_list), epoch)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/domain_loss", np.mean(domain_loss_list), epoch)

    epoch_loss = np.mean(loss_list)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/training_loss", epoch_loss, epoch)

    return {'avg_loss': epoch_loss}


def test_multihead(model, dataloader, criterion, epoch=-1, **kwargs):
    if kwargs.get('mets') is None:
        mets = ['corr_avg']
    else:
        mets = kwargs['mets']

    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []
    return_dict = {}

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc="Testing ... "):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)
        try:
            loss = criterion(output['output']['pmemo'].float(), labels.float())
            preds_list.append(output['output']['pmemo'].cpu().detach().numpy())
        except:
            loss = criterion(output.float(), labels.float())
            preds_list.append(output.cpu().detach().numpy())

        loss_list.append(loss.item())
        labels_list.append(labels.cpu().detach().numpy())

    epoch_test_loss = np.mean(loss_list)
    return_dict['avg_loss'] = epoch_test_loss
    if kwargs.get('compute_metrics', True) is True:
        return_dict.update(compute_metrics(np.vstack(labels_list), np.vstack(preds_list), metrics_list=mets))

    return return_dict, np.vstack(labels_list), np.vstack(preds_list)


def test(model, dataloader, criterion, epoch=-1, **kwargs):
    if kwargs.get('mets') is None:
        mets = ['corr_avg']
    else:
        mets = kwargs['mets']

    if kwargs.get('test_output') is not None:
        test_output = kwargs['test_output']
    else:
        test_output = 'output'

    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []
    return_dict = {}

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc=f"Testing {test_output} ... "):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)[test_output]

        if (idxs := kwargs.get('idxs')) is not None:
            output = output[:, idxs[0]: idxs[1]]

        loss = criterion(output.float(), labels.float())
        preds_list.append(output.cpu().detach().numpy())

        if kwargs.get('loss_batch_average'):
            loss = torch.mean(torch.sum(loss, axis=-1))
        loss_list.append(loss.item())
        labels_list.append(labels.cpu().detach().numpy())

    epoch_test_loss = np.mean(loss_list)
    return_dict['avg_loss'] = epoch_test_loss
    if kwargs.get('compute_metrics', True) is True:
        return_dict.update(compute_metrics(np.vstack(labels_list), np.vstack(preds_list), metrics_list=mets))

    return return_dict, np.vstack(labels_list), np.vstack(preds_list)


def predict_mls(model, dataloader, mls):
    model.eval()
    preds_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc='Predicting...'):
        song_ids, inputs = batch
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(len(inputs.shape) - 2)

        output = model(inputs)

        ml_preds = output['output'].cpu().detach().numpy()
        preds_list.append(np.hstack([np.array(song_ids).reshape((len(song_ids), 1)), ml_preds]))

    preds_np = np.vstack(preds_list)
    preds = pd.DataFrame(preds_np, columns=['path'] + mls)

    return preds


def predict_tags(model, dl, mlb, **kwargs):
    model.eval()

    preds_list = []
    labels_list = []

    for batch_idx, batch in tqdm(enumerate(dl), ascii=True, total=len(dl), desc=f"Predicting ... "):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)['output']

        if (idxs := kwargs.get('idxs')) is not None:
            output = output[:, idxs[0]: idxs[1]]

        preds_list.append(output.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    return np.vstack(labels_list), np.vstack(preds_list)


def predict_multi_model(models, dataloader, mls, aggregate='random'):
    logger.info(f"Multi model predict with {len(models)} models and {aggregate} aggregate...")
    for model in models:
        model.eval()
    preds_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc='Predicting...'):
        song_ids, inputs = batch
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(len(inputs.shape) - 2)

        if aggregate == 'random':
            model = np.random.choice(models)
            output = model(inputs)['output']

        elif aggregate == 'average':
            outs = []
            for model in models:
                outs.append(model(inputs)['output'])
            output = torch.mean(torch.stack(outs, dim=0), dim=0)
        else:
            raise Exception(f"aggregate must be in ['random, 'average'] - is {aggregate}")

        ml_preds = output.cpu().detach().numpy()
        preds_list.append(np.hstack([np.array(song_ids).reshape((len(song_ids), 1)), ml_preds]))

    preds_np = np.vstack(preds_list)
    preds = pd.DataFrame(preds_np, columns=['path'] + mls)

    return preds


def get_effects(model, dataloader, **kwargs):
    preds_list = []
    concepts_list = []
    labels_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        model_out = model(inputs)

        output = model_out['output']
        concepts = model_out['concepts']

        if (idxs := kwargs.get('idxs')) is not None:
            output = output[:, idxs[0]: idxs[1]]
            concepts = concepts[:, idxs[0]: idxs[1]]

        preds_list.append(output.cpu().detach().numpy())
        concepts_list.append(concepts.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    return {'concepts': np.vstack(concepts_list), 'output': np.vstack(preds_list), 'labels': np.vstack(labels_list)}


class ReduceOnPlateau(ReduceLROnPlateau):
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logger.info('Epoch {:5d}: reducing learning rate'
                                ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
