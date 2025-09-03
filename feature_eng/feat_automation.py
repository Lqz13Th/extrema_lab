from datetime import date

from feature_eng.feat_resample import *

def feat_auto_process(
        feat_init_date: str,
        threshold: float = 0.0013,
        resample: bool = True,
        start_symbol: str | None = None,
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

    if  start_symbol and start_symbol in symbols_automation_list_usdt:
        start_idx = symbols_automation_list_usdt.index(start_symbol)
        symbols_automation_list_usdt = symbols_automation_list_usdt[start_idx:]
    else:
        print(f"{start_symbol} not found in symbol list, running from start.")

    data_resampling(
        start_date=feat_init_date,
        end_date=last_month_end_date_str,
        threshold=threshold,
        output_dir=feat_output_dir,
        target_instruments=symbols_automation_list_usdt,
        resample=resample,
    )

if __name__ == "__main__":
    init_date = "2024-10-01"
    # feat_auto_process(feat_init_date=init_date, threshold=0.0013, resample=True)
    # feat_auto_process(feat_init_date=init_date, threshold=0.0031, resample=True)
    feat_auto_process(feat_init_date=init_date, threshold=0.0067, resample=True)
    # feat_auto_process(feat_init_date=init_date, threshold=0.013, resample=True, start_symbol="DOGEUSDT")
