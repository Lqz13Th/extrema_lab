from data_proc.data_automation import *
from feature_eng.feat_automation import *

# automation all
if __name__ == "__main__":
    init_date = "2025-01-01"
    data_auto_process(init_date)
    feat_auto_process(init_date)