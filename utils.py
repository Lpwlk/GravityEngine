import os
import platform
import time

from rich import inspect
from rich.box import ROUNDED
from rich.console import Console
from rich.progress import (Progress, ProgressBar, SpinnerColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.traceback import install
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from theme import engine_theme

install()

console = Console(
    theme = engine_theme,
    
)
console._log_render.omit_repeated_times = False
console._log_render.show_path = False

progress = Progress(
    SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), 
    console = console, 
    transient = False
)

def log(*args, t:int = .2, **kwargs):
    console.log(*args, **kwargs)
    time.sleep(t)

def header(split: bool = False) -> Table:
    header = Table(
        title = f'Init header',
        title_justify = 'left',
        border_style = 'reset',
        box = ROUNDED,
    )
    header.add_column('Execution infos', style = 'cornflower_blue', header_style = 'bold cornflower_blue')
    header.add_column('Variable states', style = 'cornflower_blue', header_style = 'bold cornflower_blue')
    header.add_row('Node username', f'{platform.uname().node}')
    header.add_row('CPU architecture', f'{platform.uname().machine}')
    header.add_row('Operating system', f'{platform.uname().system}', end_section = split)
    header.add_row('Python version',   f'{platform.python_version()}')
    header.add_row('Module name', f'{os.path.basename(__file__)}')
    log(header, justify = 'left')
