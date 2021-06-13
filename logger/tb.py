from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import os


def dict_to_table(dict):
    lens = [len(str(dict[k])) if len(str(dict[k])) > len(k) else len(str(k)) for k in dict]

    strs = ["{{: ^{}}}".format(l) for l in lens]
    underline = ["-"*l for l in lens]
    strss = "  ".join(strs)
    underline = "".join(underline)

    keys = [str(k) for k in dict.keys()]
    vals = [str(v) for v in dict.values()]
    lines = "    " + strss.format(*keys) + "\n" + "--- " + underline + "\n" + "    " + strss.format(*vals)
    return lines


def write_args(args, logger):
    writer = logger.writer
    writer.add_text('args', dict_to_table(vars(args)))


class Logger:
    def __init__(self, dir, version=None):
        """ Logger class to handle tensorboard logging

        Args:
            dir (str): Directory to log to
            version (int/str): Current version of the run. If None will be infered
        """
        self.version = version if version is not None else self.guess_version(dir)
        self.summary_writer = SummaryWriter(os.path.join(dir, 'version_{}'.format(self.version)))
        self.log_txt = os.path.join(dir, 'version_{}'.format(self.version), 'log_metrics.txt')

    def guess_version(self, dir):
        os.makedirs(dir, exist_ok=True)
        dirs = os.listdir(dir)

        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            return "{}-{}".format(os.environ['SLURM_ARRAY_JOB_ID'], os.environ['SLURM_ARRAY_TASK_ID'])

        cur_version = 0
        for d in dirs:
            if 'version_' in d:
                _, vers = d.split('_')
                if int(vers) >= int(cur_version):
                    cur_version = int(vers) + 1
        return cur_version

    def write_dict(self, vals, global_step=0):
        for k, v in vals.items():
            # print('k:', k)
            # print('v:', v)
            with open(self.log_txt, 'a') as f:
                f.write(str(k) + ' : ' + str(v.cpu().numpy().round(4)) + '\n')
            self.writer.add_scalar(k, v, global_step=global_step)

    @property
    def writer(self):
        return self.summary_writer

    @property
    def log_dir(self):
        return self.summary_writer.log_dir

    def get_version(self):
        return self.version


class ParseTb:
    def __init__(self, log_file):
        """ Helper object to parse tensorboard logs

        Args:
            log_file (str): Log dir to parse
        """
        if 'events.out' not in log_file:
            for f in os.listdir(log_file):
                if 'events.out' in f:
                    log_file = os.path.join(log_file, f)
                    break

        self.ea = event_accumulator.EventAccumulator(log_file,
                                                size_guidance={
                                                    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                    event_accumulator.IMAGES: 4,
                                                    event_accumulator.AUDIO: 4,
                                                    event_accumulator.SCALARS: 0,
                                                    event_accumulator.HISTOGRAMS: 1,
                                                })
        self.ea.Reload()
        self.scalar_tags = []
        self.text_tags = []

    def add_scalars(self, scalar_list):
        """
        Args:
            scalar_list (list[str]): List of scalar tags to retrieve on parse
        """
        self.scalar_tags = scalar_list
        return self

    def add_text(self, text_list):
        """
        Args:
            text_list (list[str]): List of text tags to retrieve on parse
        """
        self.text_tags = text_list
        return self

    def grab_hist(self, tag):
        return self.ea.Histograms(tag)

    def grab_img(self, tag):
        return self.ea.Images(tag)

    def parse(self, metadata=False):
        """ Parses the log to retrieve tags

        Args:
            metadata (bool): If true, also return scalar metadata

        Returns (tuple(dict, dict)): (scalars, text)
        """
        scalars = {st: self.ea.Scalars(st) for st in self.scalar_tags}
        text = {tt: self.ea.Tensors(tt+'/text_summary') for tt in self.text_tags}

        if not metadata:
            scalars = {tag: [d.value for d in scalar] for tag, scalar in scalars.items()}

        return scalars, text

