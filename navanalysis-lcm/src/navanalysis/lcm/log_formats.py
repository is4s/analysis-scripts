class fmts:
    """Formats for printing to terminal"""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    DRKGRAY = '\033[90m'
    LTGRAY = '\033[37m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    TAN = '\033[2m'
    WHITE = '\033[97m'


DEBUG = f'{fmts.OKBLUE}[DEBUG]{fmts.ENDC}'
INFO = f'{fmts.OKGREEN}[INFO]{fmts.ENDC}'
WARN = f'{fmts.WARNING}[WARN]{fmts.ENDC}'
ERROR = f'{fmts.FAIL}[ERROR]{fmts.ENDC}'
