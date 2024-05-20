import sys
import os

from termcolor import colored
from configs import cfg
from utils import custom_print

class Logger(object):
    r"""Duplicates all stdout to a file."""
    def __init__(self):
        path = os.path.join(cfg.logdir, 'logs.txt')

        log_dir = cfg.logdir
        if not cfg.resume and os.path.exists(log_dir):
            if cfg.clear:
                custom_print(colored('remove contents of directory %s' % log_dir, 'red'))
                os.system('rm -r %s/*' % log_dir)
            else:
                user_input = input(f"log dir \"{log_dir}\" exists. \nRemove? (y/n):")
                if user_input == 'y':
                    custom_print(colored('remove contents of directory %s' % log_dir, 'red'))
                    os.system('rm -r %s/*' % log_dir)
                else:
                    custom_print(colored('exit from the training.', 'red'))
                    exit(0)
        # this would bring error for ddp mode
        # if not os.path.exists(log_dir):
        #     os.makedirs(cfg.logdir)
        os.makedirs(cfg.logdir, exist_ok=True)

        self.log = open(path, "a") if os.path.exists(path) else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def print_config(self):
        custom_print("\n\n######################### CONFIG #########################\n")
        custom_print(cfg)
        custom_print("\n##########################################################\n\n")
