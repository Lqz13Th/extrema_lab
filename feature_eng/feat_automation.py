from datetime import date

from feature_eng.feat_resample import *

def feat_auto_process(
        init_date: str,
        threshold: float = 0.0013,
        rolling_window: int = 500,
        resample: bool = True,
):
    feat_output_dir = "../data_proc/resampled_data"

    today = date.today()
    first_day_this_month = today.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    last_month_end_date_str = last_day_last_month.strftime("%Y-%m-%d")
    print(last_day_last_month)

    with open("../symbols.json", "r", encoding="utf-8") as f_automation:
        symbols_automation_list = json.load(f_automation)

    symbols_automation_list_usdt = [s if s.endswith("T") else s + "T" for s in symbols_automation_list]
    symbols_automation_list_usdt = ["BTCUSDT"]
    data_resampling(
        start_date=init_date,
        end_date=last_month_end_date_str,
        threshold=threshold,
        rolling_window=rolling_window,
        output_dir=feat_output_dir,
        target_instruments=symbols_automation_list_usdt,
        resample=resample,
    )

if __name__ == "__main__":
    init_date = "2025-01-01"
    feat_auto_process(init_date=init_date, threshold=0.0013, rolling_window=100, resample=True)
    feat_auto_process(init_date=init_date, threshold=0.0013, rolling_window=200, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0013, rolling_window=500, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0013, rolling_window=1000, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0013, rolling_window=2000, resample=False)

    feat_auto_process(init_date=init_date, threshold=0.0031, rolling_window=100, resample=True)
    feat_auto_process(init_date=init_date, threshold=0.0031, rolling_window=200, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0031, rolling_window=500, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0031, rolling_window=1000, resample=False)
    feat_auto_process(init_date=init_date, threshold=0.0031, rolling_window=2000, resample=False)

    df = pl.read_parquet("../data_proc/resampled_data/BTCUSDT_merged_thr0.0013_roll500.parquet")
    print(df)