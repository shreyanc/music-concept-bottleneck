import json
import os
import cProfile

from helpers.loss_funcs import ScaledCriterion
# from models.model_utils import print_model_summary

pr = cProfile.Profile()
import torch.utils.data
from torch import optim, nn
import torch_optimizer as torchoptim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from datasets.midlevel import load_midlevel_aljanaki, MidlevelDataset
from datasets.mtgjamendo import load_mtgjamendo_dfs, MTGJamendoDataset
from helpers.losses import FocalLoss
from helpers.specaugment import SpecAugment, transform_random_roll_time, SpecTransform, mask_along_axis_iid
from helpers.train_funcs import train, test, train_explainable_mixed_batches_var_loss, train_explainable_mixed_batches
from imports import *
from models.cpresnet import CPResnet
from models.cpresnet_emo import CPResnetExplainable
from models.model_configs import config_cp_field_shallow_m2
from paths import *
from utils import *
from experiment import Experiment, args_override
import composer.functional as cf

import wandb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_valid_split = train_test_split

num_labels_dict = {'ml': 7, 'ins': 41, 'mood': 56, 'genre': 95, 'deam': 2, 'deam+pmemo': 2}

wandb_group_name = None
wandb_activated = True


def init_logger(run_dir, run_name):
    global logger
    fh = logging.FileHandler(os.path.join(run_dir, f'{run_name}.log'))
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)


def get_es_metric_target(m):
    if m in ['prauc', 'r2', 'rocauc']:
        return 'maximize'
    if m in ['rmse', 'avg_loss']:
        return 'minimize'


def get_main_dataset(dset_name, seed, audio_len, save_ids_dir, **kwargs):
    if dset_name == 'mood':
        tr_df, va_df, te_df, mlb = load_mtgjamendo_dfs(seed=seed, mediaeval=True, trainset_fraction=kwargs.get('mood_dset_fraction'))
        # tr_df, va_df, te_df, mlb = load_mtgjamendo_dfs(seed=seed, mediaeval=False, trainset_fraction=kwargs.get('mood_dset_fraction'))
        mlb_mood = mlb['MOOD']
        tr_dset = MTGJamendoDataset(annotations=tr_df, binarizer=mlb_mood, subset='MOOD', duration=audio_len, normalize_inputs='single')
        va_dset = MTGJamendoDataset(annotations=va_df, binarizer=mlb_mood, subset='MOOD', duration=audio_len, normalize_inputs='single')
        te_dset = MTGJamendoDataset(annotations=te_df, binarizer=mlb_mood, subset='MOOD', duration=audio_len, normalize_inputs='single')
        pickledump(tr_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_mood_tr_track_ids.pkl'))
        pickledump(va_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_mood_va_track_ids.pkl'))
        pickledump(te_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_mood_te_track_ids.pkl'))

    elif dset_name == 'genre':
        tr_df, va_df, te_df, mlb = load_mtgjamendo_dfs(seed=seed, subsets=['GENRE'])
        mlb_genre = mlb['GENRE']
        tr_dset = MTGJamendoDataset(annotations=tr_df, binarizer=mlb_genre, subset='GENRE', duration=audio_len, normalize_inputs='single')
        va_dset = MTGJamendoDataset(annotations=va_df, binarizer=mlb_genre, subset='GENRE', duration=audio_len, normalize_inputs='single')
        te_dset = MTGJamendoDataset(annotations=te_df, binarizer=mlb_genre, subset='GENRE', duration=audio_len, normalize_inputs='single')
        pickledump(tr_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_genre_tr_track_ids.pkl'))
        pickledump(va_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_genre_va_track_ids.pkl'))
        pickledump(te_df['TRACK_ID'].values, os.path.join(save_ids_dir, 'mtg_genre_te_track_ids.pkl'))

    else:
        raise ValueError(f"dset_name not understood. Can be in ['mood', 'genre'], is {dset_name}")

    return tr_dset, va_dset, te_dset


def get_loss_functions(tasks_list, **kwargs):
    loss_funcs = {}
    if kwargs.get('loss_batch_average'):
        reduction = 'none'
    else:
        reduction = 'mean'

    for task_name in tasks_list:
        if task_name in ['mood', 'genre', 'ins']:
            if kwargs[f'{task_name}loss'] == 'focal':
                logger.info(f"Selected FOCAL Loss for {task_name}")
                loss_funcs[task_name] = FocalLoss(alpha=0.25, gamma=2.0).to(device)
            else:
                logger.info(f"Selected BCE Loss for {task_name}")
                loss_funcs[task_name] = nn.BCEWithLogitsLoss(reduction=reduction).to(device)

        if task_name == 'deam':
            loss_funcs[task_name] = nn.MSELoss().to(device)

        if task_name == 'deam+pmemo':
            loss_funcs[task_name] = nn.MSELoss().to(device)

        if task_name == 'ml':
            loss_funcs[task_name] = nn.MSELoss(reduction=reduction).to(device)

    return loss_funcs


def dset_to_loader(dset, bs):
    num_workers = 1 if os.uname()[1] in ['shreyan-HP-EliteBook-840-G5', 'shreyan-All-Series'] else bs
    return DataLoader(dset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)


def main(h, run_num, seed, exp_params):
    debug = exp_params['debug']
    # WANDB SETUP =============================================
    if not wandb_activated or debug or os.uname()[1] in ['shreyan-HP-EliteBook-840-G5', 'shreyan-All-Series']:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"

    if wandb_sweep:
        run = wandb.init(config=h, reinit=True, mode=wandb_mode)
        short_run_name = run.name
        exp_params['run_name'] = run.name
        exp_params['run_dir'] = os.path.join(exp_params['sweep_dir'], run.name)
        if not os.path.exists(exp_params['run_dir']):
            os.makedirs(exp_params['run_dir'])
        init_logger(exp_params['run_dir'], f"{exp_params['sweep_name']}_{exp_params['run_name']}")

    else:
        short_run_name = exp_params['run_name'][:5]
        global wandb_group_name
        if wandb_group_name is None:
            wandb_group_name = short_run_name
            wandb_run_name = f"{short_run_name}_{run_num}"
        else:
            wandb_run_name = f"{short_run_name}_{run_num}_{wandb_group_name}"
        run = wandb.init(project=exp_name, group=wandb_group_name, name=wandb_run_name, config=h, reinit=True, mode=wandb_mode)
    # =========================================================
    datalogger = DataLogger(os.path.join(exp_params['run_dir'], 'datalogs', f'run_{run_num}'))
    logger.info(f"===================================================")
    logger.info(f"Running {' '.join(sys.argv)}")
    logger.info(f"with pid {os.getpid()}")

    logger.info(f"Running main func with seed = {seed}")
    logger.debug("Debug print")
    seed_everything(seed)

    with open(os.path.join(exp_params['run_dir'], 'config.json'), 'w') as f:
        json.dump(h, f)

    if h['explainable']:
        if h.get('concepts') == 'ml':
            concepts = ['ml']
        elif h.get('concepts') == 'ins':
            concepts = ['ins']
        elif h.get('concepts') == 'ml+ins':
            concepts = ['ml', 'ins']
        else:
            logger.info(f"Selected default concetps: ml")
            concepts = ['ml']
        logger.info(f"Concepts: {concepts}")

    # model_arch_config = {k: h[k] for k in ('concept_input_bias', 'concept_input_bn', 'concept_layer_type', 'ff_bias')}
    # model_arch_config = {k: h[k] for k in ('concept_input_bn', 'ff_bias')}


    audio_length = h['audlen']

    if h['augment']:
        logger.info("Augment = TRUE")
        augmenter = SpecAugment([
            SpecTransform(transform_random_roll_time),  # random slide in time dimension
            SpecTransform(mask_along_axis_iid, axis=(0, 0), p=(0.0, 0.3)),  # freq mask
            SpecTransform(mask_along_axis_iid, axis=(1, 1), p=(0.0, 0.3)),  # time mask
        ])
    else:
        augmenter = None

    # INIT DATA
    tr_dataset, va_dataset, te_dataset = get_main_dataset(h['maintask'], seed=seed, audio_len=audio_length, save_ids_dir=exp_params['run_dir'],
                                                          mood_dset_fraction=h.get('moodfraction'))

    concept_ml_tr_ids, concept_ml_te_ids = load_midlevel_aljanaki(tsize=0.1)
    concept_ml_tr_dataset = MidlevelDataset(select_song_ids=concept_ml_tr_ids, duration=audio_length, normalize_inputs='single')
    concept_ml_te_dataset = MidlevelDataset(select_song_ids=concept_ml_te_ids, duration=audio_length, normalize_inputs='single')

    concept_ins_tr_df, concept_ins_va_df, concept_ins_te_df, mlb = load_mtgjamendo_dfs(seed=seed, mediaeval=False, subsets=['INSTRUMENT'])
    mlb_ins = mlb['INSTRUMENT']
    concept_ins_tr_dataset = MTGJamendoDataset(annotations=concept_ins_tr_df, binarizer=mlb_ins, subset='INSTRUMENT', duration=audio_length, normalize_inputs='single')
    concept_ins_va_dataset = MTGJamendoDataset(annotations=concept_ins_va_df, binarizer=mlb_ins, subset='INSTRUMENT', duration=audio_length, normalize_inputs='single')
    concept_ins_te_dataset = MTGJamendoDataset(annotations=concept_ins_te_df, binarizer=mlb_ins, subset='INSTRUMENT', duration=audio_length, normalize_inputs='single')

    if debug or wandb_sweep:
        tr_dataset = pytorch_random_sampler(tr_dataset, num_samples=5000)
        concept_ins_tr_dataset = pytorch_random_sampler(concept_ins_tr_dataset, num_samples=5000)
        concept_ml_te_dataset = pytorch_random_sampler(concept_ml_te_dataset, num_samples=100)

    logger.info(f"LENGTHS: tr={len(tr_dataset)}, va={len(va_dataset)}, te={len(te_dataset)}")

    if h['explainable']:
        if 'ml' in concepts:
            logger.info(f"LENGTHS (Midlevel): ml_tr={len(concept_ml_tr_dataset)}, ml_te={len(concept_ml_te_dataset)}")
        if 'ins' in concepts:
            logger.info(f"LENGTHS (MTG Instruments) ins_tr={len(concept_ins_tr_dataset)}, ins_te={len(concept_ins_te_dataset)}")

    tr_dataloader = dset_to_loader(tr_dataset, bs=h['bs'])
    va_dataloader = dset_to_loader(va_dataset, bs=h['bs'])
    te_dataloader = dset_to_loader(te_dataset, bs=h['bs'])

    concept_ml_tr_dataloader = dset_to_loader(concept_ml_tr_dataset, bs=h['bs'])
    concept_ml_te_dataloader = dset_to_loader(concept_ml_te_dataset, bs=h['bs'])

    concept_ins_tr_dataloader = dset_to_loader(concept_ins_tr_dataset, bs=h['bs'])
    concept_ins_te_dataloader = dset_to_loader(concept_ins_te_dataset, bs=h['bs'])

    model_arch_config = {'ff_hidden_size': h['ff_hidden_size']}

    if h['explainable']:
        num_concepts = sum([v for k, v in num_labels_dict.items() if k in concepts])
        net = CPResnetExplainable(num_interpretable_concepts=num_concepts + h['extranodes'], num_targets=num_labels_dict[h['maintask']], **model_arch_config).to(device)
        # net = CPResnetExplainableMLInsSigmoid(num_interpretable_concepts=num_concepts + h['extranodes'], num_targets=num_labels_dict[h['maintask']], **model_arch_config).to(device)
    else:
        if h['network'] == 'bottleneck':
            assert h['bottlenecksize'] > 0
            net = CPResnetExplainable(num_interpretable_concepts=h['bottlenecksize'], num_targets=num_labels_dict[h['maintask']], **model_arch_config).to(device)
        elif h['network'] == 'baseline':
            net = CPResnet(num_targets=num_labels_dict[h['maintask']], config=config_cp_field_shallow_m2).to(device)
        else:
            raise ValueError(f"Invalid network name ({h['network']})")

    net = cf.apply_blurpool(net).to(device)
    net = cf.apply_squeeze_excite(net).to(device)
    # net = cf.apply_factorization(net)
    # net = cf.apply_ghost_batchnorm(net, ghost_batch_size=8)
    # print_model_summary(net)

    model_name = net.name


    if h['explainable']:
        tasks = [h['maintask']] + concepts
    else:
        tasks = [h['maintask']]
    criteria_dict = get_loss_functions(tasks, **h)
    criteria_list = [criteria_dict[t] for t in tasks]
    # scaled_crit = ScaledCriterion(criteria_list, [0.58, 0.05, 0.37])
    scaled_crit = ScaledCriterion(criteria_list, [0.8, 0.2])

    params = ([p for p in net.parameters()])

    if h['optimizer'] == 'adam':
        optimizer = optim.Adam(params, lr=h['lr'], weight_decay=h['adam_wd'], betas=(h['adam_b1'], h['adam_b2']))

    if h['scheduler'] == 'multistep':
        logger.info("Multi step scheduler selected")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 10, 15, 20], gamma=0.5)
    elif h['scheduler'] == 'cosine':
        logger.info("Cosine scheduler selected")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=h['cosine_t0'], T_mult=h['cosine_t_mult'])

    task_metrics = {
        'deam': ['r2', 'rmse'],
        'deam+pmemo': ['r2', 'rmse'],
        'mood': ['prauc', 'rocauc'],
        'genre': ['prauc', 'rocauc']
    }

    es_metrics = {
        'deam': 'rmse',
        'deam+pmemo': 'rmse',
        'mood': 'prauc',
        'genre': 'prauc',
    }

    es_metric = es_metrics[h['maintask']]
    es_condition = get_es_metric_target(es_metric)

    es = EarlyStopping(patience=h['patience'], condition=es_condition, verbose=True,
                       save_dir=os.path.join(exp_params['run_dir'], 'saved_models'),
                       saved_model_name=model_name + '_' + short_run_name + f"_{run_num}")

    # if h['plateau']:
    #     logger.info("Reduce LR on Pleateau selected")
    #     scheduler_plateau = ReduceLROnPlateau(optimizer, get_es_metric_target(es_metric)[:3], factor=h['plateau_factor'],
    #                                           patience=h['plateau_patience'], cooldown=h['plateau_cooldown'], verbose=True)

    # if not (h['plateau'] or h['scheduler']):
    #     logger.info("NO scheduler selected")

    num_cycles = 3
    num_epochs = int(h['cosine_t0'] * ((h['cosine_t_mult']**num_cycles - 1)/(h['cosine_t_mult']-1)) + 1)
    es_start_epoch = num_epochs - h['patience'] + 1

    # load_model('/home/shreyan/mounts/home@fs/RUNS/thesis_cbm_genre_weighted/f05ec_2022-07-06_02-20-34_genre_0.58-0.05-0.37_audlen=15_extranodes=1/saved_models/CPResnetExplainableMLInsSigmoid_f05ec_0.pt', net)

    for epoch in range(1, num_epochs if not debug else 2):
        wandb.log({"Epoch": epoch})
        if h['explainable']:
            used_dloaders = [tr_dataloader]
            if 'ml' in concepts:
                used_dloaders.append(inf(concept_ml_tr_dataloader))
            if 'ins' in concepts:
                used_dloaders.append(inf(concept_ins_tr_dataloader))

            mixed_dataloader = zip(*used_dloaders)

            tr_losses = train_explainable_mixed_batches_var_loss(
                net,
                mixed_dataloader,
                optimizer,
                tasks,
                scaled_crit,
                run_name=f"{exp_params['run_name']}_epoch-{epoch}",
                epoch=epoch,
                dataloader_len=len(tr_dataloader),
                loss_batch_average=h.get('loss_batch_average')
            )

            wandb.log({f"loss/train_{k}": v for k, v in tr_losses.items()})

            if 'ml' in concepts:
                concept_ml_metrics = test(
                    net,
                    concept_ml_te_dataloader,
                    criteria_dict['ml'],
                    epoch,
                    mets=['corr_avg', 'r2'],
                    test_output='concepts',
                    idxs=(0, 7),
                    loss_batch_average=h.get('loss_batch_average')
                )[0]
                logger.info(f"Epoch {epoch} concept_ml corr = {concept_ml_metrics['corr_avg']:.4f}")
                wandb.log({"ml_corr": concept_ml_metrics['corr_avg'], "ml_r2": concept_ml_metrics['r2']})

            if 'ins' in concepts:
                concept_ins_metrics = test(
                    net,
                    concept_ins_te_dataloader,
                    criteria_dict['ins'],
                    epoch,
                    mets=['prauc', 'rocauc'],
                    test_output='concepts',
                    idxs=(7, 48) if h['concepts'] == 'ml+ins' else None,  # TODO: Fix this
                    loss_batch_average=h.get('loss_batch_average')
                )[0]
                logger.info(f"Epoch {epoch} concept_ins prauc = {concept_ins_metrics['prauc']:.4f}")
                wandb.log({"ins_prauc": concept_ins_metrics['prauc'], "ins_rocauc": concept_ins_metrics['rocauc']})

                # if h.get('extranodes') > 0:
                #     try:
                #         summed_abs_weights = torch.abs(net.feed_forward.weight).sum(dim=0)
                #     except AttributeError:
                #         summed_abs_weights = torch.abs(net.module.feed_forward.weight).sum(dim=0)
                #
                #     concept_weights_mean = summed_abs_weights[:num_concepts].mean()
                #     extranodes_weight_mean = summed_abs_weights[num_concepts:].mean()
                #     wandb.log({'mean_concept_weight': concept_weights_mean, 'mean_extranodes_weight': extranodes_weight_mean})

        else:
            tr_losses = train(
                net,
                tr_dataloader,
                optimizer,
                criteria_dict[h['maintask']],
                epoch,
                exp_params['run_name'] + f'_epoch-{epoch}',
                dataloader_len=len(tr_dataloader)
            )
            wandb.log({"loss/train": tr_losses['avg_loss']})

        val_metrics = test(net, va_dataloader, criteria_dict[h['maintask']], epoch, mets=task_metrics[h['maintask']], loss_batch_average=h.get('loss_batch_average'))[0]
        wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
        wandb.log({f"loss/val": val_metrics['avg_loss']})

        logger.info(f"Epoch {epoch} val {es_metric} = {val_metrics[es_metric]:.4f}")

        if h.get('scheduler') in ['multistep', 'cosine']:
            scheduler.step()
        # if h.get('plateau'):
        #     scheduler_plateau.step(val_metrics[es_metric])

        if epoch > es_start_epoch:
            es(val_metrics[es_metric], net)
            if es.early_stop:
                wandb.log({f"run_val_{es_metric}": es.best_score})
                logger.info(f"Early stop - trained for {epoch - es.counter} epochs - best metric {es.best_score}")
                break

    load_model(es.save_path, net)
    #
    # final_val_metrics = test(net, va_dataloader, criteria_dict[h['maintask']], epoch=-1, mets=['prauc', 'rocauc'], loss_batch_average=h.get('loss_batch_average'))[0]
    # vm_str = [f"{m}={final_val_metrics[m]}" for m in final_val_metrics]
    # logger.info(f"Final val metrics: {vm_str}")

    final_test_metrics = test(net, te_dataloader, criteria_dict[h['maintask']], epoch=-1, mets=['prauc', 'rocauc'], loss_batch_average=h.get('loss_batch_average'))[0]
    tm_str = [f"{m}={final_test_metrics[m]}" for m in final_test_metrics]
    logger.info(f"Final test metrics: {tm_str}")

    maintask_metrics = dict(
        # run_val_prauc=final_val_metrics['prauc'],
        # run_val_rocauc=final_val_metrics['rocauc'],
        PRAUC=final_test_metrics['prauc'],
        ROCAUC=final_test_metrics['rocauc']
    )

    concept_metrics = {}
    if h['explainable']:
        concept_metrics = {}
        if 'ml' in concepts:
            concept_ml_metrics = test(net, concept_ml_te_dataloader, criteria_dict['ml'], epoch=-1, mets=['corr_avg', 'r2'], test_output='concepts', idxs=(0, 7),
                                      loss_batch_average=h.get('loss_batch_average'))[0]
            concept_metrics.update({'ml_corr': concept_ml_metrics['corr_avg'], 'ml_r2': concept_ml_metrics['r2']})
            wandb.log({"ml_corr": concept_ml_metrics['corr_avg'], "ml_r2": concept_ml_metrics['r2']})

        if 'ins' in concepts:
            concept_ins_metrics = test(net, concept_ins_te_dataloader, criteria_dict['ins'], epoch=-1, mets=['prauc', 'rocauc'], test_output='concepts', idxs=(7, 48),
                                       loss_batch_average=h.get('loss_batch_average'))[0]
            concept_metrics.update({'ins_prauc': concept_ins_metrics['prauc'], 'ins_rocauc': concept_ins_metrics['rocauc']})
            wandb.log({"ins_prauc": concept_ins_metrics['prauc'], "ins_rocauc": concept_ins_metrics['rocauc']})

    wandb.run.summary.update({**maintask_metrics, **concept_metrics})
    run.finish()

    return {**maintask_metrics, **concept_metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int)
    parser.add_argument('-suffix')
    parser.add_argument('-group', type=str)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-sweep', action='store_true')
    parser.add_argument('-no_wandb', action='store_true')
    parser.add_argument('-num_runs', type=int)
    parser.add_argument('-same_seed', action='store_true')
    parser.add_argument('-hp', nargs='*')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.no_wandb:
        wandb_activated = False

    wandb_sweep = True if args.sweep else False

    wandb_group_name = args.group


    def cleanup_hparams(hp):
        pass


    hparams = dict(
        maintask='genre',
        audlen=15,
        explainable=True,
        concepts='ml',
        extranodes=0,
        bottlenecksize=8,
        moodloss='focal',
        genreloss='focal',
        insloss='focal',
        network=None,
        moodfraction=None,
        optimizer='adam',
        fa=True,
        bs=8,
        lr=0.001,
        patience=7,
        scheduler='cosine',
        augment=False,
        cosine_t0=5,
        cosine_t_mult=3,
        # plateau=False,
        # plateau_factor=0.39,
        # plateau_patience=18,
        # plateau_cooldown=1,
        adam_wd=4e-7,
        adam_b1=0.82,
        adam_b2=0.92,
        ff_hidden_size=None,
    )

    hparams = args_override(hparams, vars(args).pop('hp'))
    cleanup_hparams(hparams)

    exp_name = f"thesis_cbm_genre_weighted_ml"

    if args.sweep:
        wandb_sweep = True
        sweep_name = '31wgh6f5'
        wandb_group_name = f'sweep_{sweep_name}'
        seed = np.random.randint(0, 1000)
        sweep_dir = os.path.join(MAIN_RUN_DIR, '_sweeps', sweep_name)
        if not os.path.exists(sweep_dir):
            os.makedirs(sweep_dir)
        main(hparams, run_num=0, seed=seed, exp_params={'sweep_dir': sweep_dir, 'sweep_name': sweep_name, 'run_name': None, 'debug': False})
    else:
        wandb_sweep = False
        wandb_group_name = args.group
        exp = Experiment(main_exp=main, hparams_exp=hparams, cli_args=args, exp_name=exp_name, override_hp=False)
        exp.run(1)
        exp.log_results()
        exp.summarize_results()
