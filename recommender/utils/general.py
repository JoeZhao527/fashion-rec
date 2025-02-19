from datetime import datetime
import pandas as pd

def log(msg):
    print(f"[{datetime.now()}]: {msg}")

def log_configuaration(**kwargs):
    log(f"System configurations:")
    print(pd.DataFrame(list(kwargs.items()), columns=["attr", "value"]))