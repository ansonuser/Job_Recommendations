from selenium.webdriver.chrome.options import Options
import os
import logging
import datetime
import yaml

def load_config(filename="data"):
    cfg_path = os.path.abspath(os.getcwd()) + f"\\..\\configs\\{filename}.yaml"
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

# for selenium running in background
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")


HEADERS= {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0.1) Gecko/2010010' \
    '1 Firefox/4.0.1',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language':'en-us,en;q=0.5',
    'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.7' 
}

def get_logger(log_path, level):
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(__name__)  # Use a unique logger n
    log_file = os.path.join(log_path, str(datetime.datetime.today().date()-datetime.timedelta(days=0))) +'.log' 
    handler = logging.FileHandler(log_file, 'a', 'utf-8')
    formatter = logging.Formatter('%(asctime)s;%(levelname)s:%(message)s',"%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(handler)

    logger.setLevel(level)  # Ensure logger respects the logging level

    return logger

CFG = load_config()
# Logs
log_path = os.path.abspath(os.getcwd()) + "\\..\\Logs"
Logger = get_logger(log_path, CFG["logger"]["level"])


