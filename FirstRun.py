import os
import pandas as pd
import sys
import logging

## modules
sys.path.insert(0, './_modules')
from path_definitions import control_dir, att_dir
from orchestra_module import mailId_query, get_table, TheGreatFilter, TheCollector, Orchestrator
from logging_module import getlogger
import random
import numpy as np
import time



logging = getlogger()


control_path = control_dir()

att_path = att_dir()


print('Running orchestrator')
results = Orchestrator(control_path, att_path, new_query = False, query_type = 'full')