"""
    Logger class 
    by: kyhoon
"""

import logging
import logging.config
import json
from setting import *
import os, sys, os.path

# for tensorboard
from tensorboardX import SummaryWriter
from pathlib import Path

# for print
from termcolor import colored

# for flush
import csv

# for debug
import pdb

class Logger:


    def __init__(self):
        self.logger = logging.Logger("root")
        log_dir = self.make_dirs()
        for handler in self.logger.handlers:
            if handler.__class__.name == 'FileHandler':
                self.writer = SummaryWriter(log_dir)
            else:
                self.writer = None
            break
        
        #log_dir = self.make_dirs()
    def read_json(self, logging_json, logger_name = "root"):
        logging.config.dictConfig(logging_json)
        self.logger = logging.getLogger(logger_name)

    def make_dirs(self):
        self.log_dir = LOG_CURRENT_PATH
        Path(self.log_dir).mkdir(parents = True, exist_ok = True)
        #self.log_dir = os.path.join(LOG_PATH, self.tag, self.name, \
        #    datetime.now()
        #self.log_dir = os.path.join(PROJECT_ROOT, LOG_DIR, self.args.tag,
        #                       datetime.now().strftime("%Y%m%d%H%M%S"))
        return self.log_dir
    
    def log(self, msg, lvl = "INFO"):
        """Print message to terminal"""
        lvl, color = self.get_level_color(lvl)
        self.logger.log(lvl, colored(msg, color))

    def add_level(self, name, lvl, color = 'white'):
        if name not in LOG_LEVELS.keys() and lvl not in LOG_LEVELS.values():
            LOG_LEVELS[name] = {'lvl': lvl, 'color': color}
            logging.addLevelName(lvl, name)
        else:
            raise AssertionError("log level already exists")

    def get_level_color(self, level):
        assert isinstance(level, str)
        global LOG_LEVELS
        level_num = LOG_LEVELS[level]['lvl']
        color = LOG_LEVELS[level]['color']
        return level_num, color
    
    def scalar_summary(self, info, step, lvl):
        assert isinstance(info, dict), "data must be a dictionary"
        self.log("logging values for step: {}".format(step), lvl)

        # flush to terminal        
        key2str = {}
        for key, val in info.items():
            if isinstance(val, float):
                valstr = "%-8.3g" %(val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if len(key2str) == 0:
            self.log("empty key-value dict", 'WARNING')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        dashes = '  ' + '-'*(keywidth + valwidth + 7)
        lines = [dashes]
        for key, val in key2str.items():
            lines.append('  | %s%s | %s%s |' %(
                key,
                ' '*(keywidth - len(key)),
                val,
                ' '*(valwidth - len(val))
            ))
        lines.append(dashes)
        print('\n'.join(lines))

        # flush to csv
        if self.log_dir is not None:
            filepath = Path(os.path.join(self.log_dir, 'values.csv'))
            if not filepath.is_file():
                with open(filepath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step'] + list(info.keys()))
            
            with open(filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([step] + list(info.values()))

        # Flush to tensorboard
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s