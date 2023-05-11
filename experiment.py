from utils import *
from paths import *
from imports import *

logger = logging.getLogger()


def args_override(exp_hps, hp_ovr):
    hp_cli = {}
    if hp_ovr is not None:
        for pair in hp_ovr:
            key, value = pair.split('=')
            hp_cli[key] = num_if_possible(value)
    exp_hps.update(hp_cli)
    return exp_hps


class Experiment:
    def __init__(self, main_exp, hparams_exp, cli_args=None, exp_name=None, override_hp=True):
        self.results = pd.DataFrame()
        self.main_func = main_exp  # Main func should accept 3 args: hyperparams_dict, run number, seed

        if override_hp:
            self.hp = args_override(hparams_exp, vars(cli_args).pop('hp'))
        else:
            self.hp = hparams_exp
        self.args = vars(cli_args)

        if self.args['seed'] is None:
            self.seed = int(int(time.time() * 10e3) - int(time.time()) * 10e3)
        else:
            self.seed = self.args['seed']

        self.run_name = self._create_run_name()
        if len(self.run_name) > 150:
            self.run_name = self.run_name[:25]

        if exp_name is None:
            self.name = __name__
        else:
            self.name = exp_name

        if self.args['debug'] or os.uname()[1] in ['shreyan-HP', 'shreyan-All-Series']:
            self.debug = True
            self.exp_dir = os.path.join(MAIN_RUN_DIR, '_debug_runs')
        else:
            self.debug = False
            self.exp_dir = os.path.join(MAIN_RUN_DIR, self.name)

        self.run_dir = os.path.join(self.exp_dir, self.run_name)

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)


        if self.debug or os.uname()[1] in ['shreyan-HP', 'shreyan-All-Series']:
            pass
        else:
            self._take_code_snapshot()

        self._init_logger()
        self.exp_params = {'run_dir': self.run_dir,
                           'run_name': self.run_name,
                           'debug': self.debug}

    def _create_run_name(self):
        dtstr = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_hash = hashlib.sha1()
        name_hash.update(str(time.time()).encode('utf-8'))
        run_hash = name_hash.hexdigest()[:5]
        run_name = f'{run_hash}_{dtstr}'
        if 'auto' in self.args['suffix']:
            run_name_str = ''
            for k, v in self.hp.items():
                run_name_str += f"_{k}={str(v)}"
            run_name += run_name_str + self.args['suffix'].replace('auto', '')
        elif self.args['suffix'] is not None:
            run_name += '_' + self.args['suffix']
        return run_name

    def _take_code_snapshot(self):
        from shutil import ignore_patterns
        shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(self.run_dir, 'code'), ignore=ignore_patterns('wandb'))

    def _init_logger(self):
        global logger
        fh = logging.FileHandler(os.path.join(self.run_dir, f'{self.run_name}.log'))
        sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)

    def run(self, times=1, change_seeds=True):
        if self.args['num_runs'] is not None:
            times = int(self.args['num_runs'])

        if self.args['same_seed']:
            change_seeds = False

        for run_num in range(times):
            logger.info(f"RUN {run_num}, SEED {self.seed}, HPs {self.hp}")
            res_dict = self.main_func(self.hp, run_num, self.seed, self.exp_params)
            self.results = self.results.append(pd.DataFrame.from_records([res_dict]), ignore_index=True)
            if change_seeds:
                self.seed += 1

    def print_results(self):
        print(self.results)

    def log_results(self):
        logger.info(f"RESULTS:\n {self.results} \n")

    def summarize_results(self, export=False, saveto=None, log=True):
        if self.results.empty:
            logger.info(f"RESULTS SUMMARY:\n <Empty Dataframe> \n")
        else:
            if export:
                self.results.agg(['count', 'mean', 'std', 'min', 'max']).to_csv(os.path.join(self.run_dir, 'results_summary.csv'))
            if saveto is not None:
                self.results.agg(['count', 'mean', 'std', 'min', 'max']).to_csv(saveto)
            if log:
                logger.info(f"RESULTS SUMMARY:\n {self.results.agg(['count', 'mean', 'std', 'min', 'max'])} \n")