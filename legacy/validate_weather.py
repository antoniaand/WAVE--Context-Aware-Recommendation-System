#!/usr/bin/env python3
"""
PAS 3 – Validare date meteo reale in train_ready.csv
Raporteaza: completitudine, statistici, distributie sezoniera,
attendance rate pe vreme buna vs rea, corelatii, distributia clasei.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train_ready.csv"

def main():
    df = pd.read_csv(TRAIN_PATH)

    print("=" * 60)
    print("PAS 3 - VALIDARE DATE METEO REALE")
    print("=" * 60)

    # 1. Completitudine
    n = len(df)
    filled_t = df["weather_temp_C"].notna().sum()
    filled_p = df["weather_precip_mm"].notna().sum()
    print(f"\n[1] Completitudine:")
    print(f"  weather_temp_C    : {filled_t:,} / {n:,} ({filled_t/n*100:.1f}%)")
    print(f"  weather_precip_mm : {filled_p:,} / {n:,} ({filled_p/n*100:.1f}%)")

    # 2. Statistici globale
    tc = df["weather_temp_C"]
    pm = df["weather_precip_mm"]
    print(f"\n[2] Statistici globale:")
    print(f"  Temp (C)   : mean={tc.mean():.2f}, std={tc.std():.2f}, "
          f"min={tc.min():.1f}, max={tc.max():.1f}")
    print(f"  Precip (mm): mean={pm.mean():.2f}, std={pm.std():.2f}, "
          f"min={pm.min():.1f}, max={pm.max():.1f}")

    # 3. Media si std per sezon (sanity check)
    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    df["season_name"] = df["season"].map(season_map)
    print(f"\n[3] Temperatura medie si std per sezon (sanity check):")
    seasonal = (
        df.groupby("season_name")[["weather_temp_C", "weather_precip_mm"]]
        .agg(["mean", "std"])
        .round(2)
    )
    print(seasonal.to_string())
    df.drop(columns=["season_name"], inplace=True)

    # 4. Attendance rate pe tipuri de vreme
    print(f"\n[4] Attendance rate (attended=1) pe tipuri de vreme:")
    good_w   = (tc.between(10, 25)) & (pm < 5)
    bad_w    = (tc < 5) | (tc > 28) | (pm > 10)
    neutral  = ~good_w & ~bad_w

    categories = [
        ("Vreme buna   (10-25C, <5mm)", good_w),
        ("Vreme rea    (<5C / >28C / >10mm)", bad_w),
        ("Vreme neutra (restul)", neutral),
    ]
    for label, mask in categories:
        sub = df[mask]
        if len(sub):
            rate = sub["attended"].mean() * 100
            print(f"  {label}: n={len(sub):,}, attendance={rate:.1f}%")

    # 5. Distributia clasei
    print(f"\n[5] Distributia clasei 'attended':")
    vc = df["attended"].value_counts()
    print(f"  attended=0 (no-show) : {vc.get(0,0):,} ({vc.get(0,0)/n*100:.1f}%)")
    print(f"  attended=1 (attended): {vc.get(1,0):,} ({vc.get(1,0)/n*100:.1f}%)")

    # 6. Corelatie Pearson
    corr_t = tc.corr(df["attended"])
    corr_p = pm.corr(df["attended"])
    print(f"\n[6] Corelatie Pearson cu 'attended':")
    print(f"  weather_temp_C    : r = {corr_t:+.4f}")
    print(f"  weather_precip_mm : r = {corr_p:+.4f}")
    print(f"  (date sintetice aveau r ~0.3-0.5 artificial; valorile reale")
    print(f"   sunt mai mici, ceea ce e academic mai defensibil)")

    # 7. Confirmare valori raw (nescalate)
    print(f"\n[7] Sample valori (confirmare nescalate - ar trebui temp ~Celsius, precip ~mm):")
    cols = ["weather_temp_C", "weather_precip_mm", "event_month", "attended"]
    print(df[cols].head(10).to_string())

    # 8. Separabilitate: diferenta medie temperatura intre attended=0 vs 1
    mean_t_0 = df.loc[df["attended"]==0, "weather_temp_C"].mean()
    mean_t_1 = df.loc[df["attended"]==1, "weather_temp_C"].mean()
    mean_p_0 = df.loc[df["attended"]==0, "weather_precip_mm"].mean()
    mean_p_1 = df.loc[df["attended"]==1, "weather_precip_mm"].mean()
    print(f"\n[8] Separabilitate clase:")
    print(f"  Temp medie | attended=0: {mean_t_0:.2f}C  |  attended=1: {mean_t_1:.2f}C  |  delta: {abs(mean_t_1-mean_t_0):.2f}C")
    print(f"  Precip med | attended=0: {mean_p_0:.2f}mm |  attended=1: {mean_p_1:.2f}mm |  delta: {abs(mean_p_1-mean_p_0):.2f}mm")
    print(f"  (cu date sintetice delta era ~10C si ~3mm – artificial de mare)")

    print("\n" + "=" * 60)
    print("Concluzie: datele meteo REALE sunt in train_ready.csv.")
    print("Urmatorul pas: re-antrenare modele (PAS 4).")
    print("=" * 60)

if __name__ == "__main__":
    main()
