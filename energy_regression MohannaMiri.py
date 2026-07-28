#----------------------------------------------------------------------------------
# first project Owtana tech Mohanna Miri
# ---------------------------------------------------------------------------------
"""
 hazfe dade haye part (Outlier) va peida kardan e behtarin
moadele khat (regression khati chandgane) baraye 2 model energy:
    1) Masraf bargh  ~  CDD + Tolid
    2) Masraf gaz    ~  HDD + Tolid

Nokte mohem: chon har model 2 ta motaghayer mostaghel dare (CDD/HDD va Tolid),
moadele khat dar vaghe ye "safhe" hast, na ye khat sade ru nemoodar 2 boodi:
    Masraf = a * (CDD ya HDD) + b * Tolid + c
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1) Khandan va tamizkari e dade az file excel
# ----------------------------------------------------------------------
# Masire kamele file (ba harf r avalesh )
FILE = r"C:\Users\User\Documents\Owtana projects\first project 3 mordad 1405.xlsx"

def load_data(path):
    # 2 radife avale file, sar sotoon vahed hastan na dade va nadide migirim -> skiprows=2
    df = pd.read_excel(
        path, skiprows=2, header=None,
        names=["tarikh", "bargh", "gaz", "CDD", "tolid", "HDD"]
    )
    # Tabdile hame sotoon haye adadi be float (age reshte ya khali bood NaN mishe)
    for c in ["bargh", "gaz", "CDD", "tolid", "HDD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 2) Tabe asli: regression khati chandgane + hazfe tekrarshavande outlier ha
# ----------------------------------------------------------------------
def fit_with_outlier_removal(df, x_cols, y_col, z_thresh=2.5, verbose=True):
    """
    x_cols: list e esm e setoon haye motaghayer mostaghel, masalan ['CDD', 'tolid']
    y_col : esm e setoon e motaghayer vabaste, masalan 'bargh'
    z_thresh: astane e residual e standard shode baraye shenasayi e outlier (mamulan 2 ta 3)

    Ravesh e kar:
      1. Model ro ba tamame dade ha fit mikonim.
      2. Residual (khataye) har noghte ro hesab va standard mikonim (shabih z-score).
      3. Har noghte i ke residuale standardesh az z_thresh bishtar bashe ro hazf mikonim.
      4. Ba dade baghimande dobare fit mikonim va in kar ro tekrar mikonim
         ta digar hich outlieri peida nashe (ya dade kheili kam beshe).
    """
    work = df.copy()
    removed_rows = []
    iteration = 0

    while True:
        iteration += 1
        X = work[x_cols].values
        y = work[y_col].values

        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        resid = y - pred

        # residuale standard shode = (khata - miangin e khata) / enheraf meyar e khata
        std_resid = (resid - resid.mean()) / resid.std(ddof=1)

        is_outlier = np.abs(std_resid) > z_thresh

        if not is_outlier.any() or work.shape[0] <= len(x_cols) + 3:
            # outlieri baghi namande, ya dade baraye hazf bishtar kheili kam shode -> tavaghof
            break

        # noghte(haye) part ro negah midarim ke print konim, bad hazfeshoon mikonim
        removed_rows.append(work.loc[is_outlier, ["tarikh", y_col] + x_cols].assign(
            std_residual=std_resid[is_outlier], iteration=iteration
        ))
        work = work.loc[~is_outlier].reset_index(drop=True)

        if verbose:
            print(f"  Tekrar {iteration}: {is_outlier.sum()} noghte part hazf shod "
                  f"(baghimande: {work.shape[0]} noghte)")

    # Fit nahaei ru dade tamiz shode
    X = work[x_cols].values
    y = work[y_col].values
    final_model = LinearRegression().fit(X, y)
    r2 = final_model.score(X, y)

    removed_df = pd.concat(removed_rows) if removed_rows else pd.DataFrame()
    return final_model, work, removed_df, r2


def print_equation(model, x_cols, y_name, r2):
    terms = " + ".join(f"({coef:.4f} x {name})" for coef, name in zip(model.coef_, x_cols))
    print(f"\nMoadele nahaei:")
    print(f"  {y_name} = {terms} + ({model.intercept_:.2f})")
    print(f"  R^2 = {r2:.4f}")


# ----------------------------------------------------------------------
# 3) Ejraye 2 model
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(FILE)
    print(f"Tedade kole dade ha: {df.shape[0]}")

    # ---------- Model 1: Bargh bar hasbe CDD va Tolid ----------
    print("\n" + "=" * 60)
    print("Model 1: Masraf bargh ~ CDD + Tolid")
    print("=" * 60)
    model_bargh, clean_bargh, removed_bargh, r2_bargh = fit_with_outlier_removal(
        df, x_cols=["CDD", "tolid"], y_col="bargh", z_thresh=2.5
    )
    print_equation(model_bargh, ["CDD", "Tolid"], "Masraf bargh", r2_bargh)
    if not removed_bargh.empty:
        print("\nNoghat e part e hazf shode:")
        print(removed_bargh.to_string(index=False))

    # ---------- Model 2: Gaz bar hasb HDD va Tolid ----------
    print("\n" + "=" * 60)
    print("Model 2: Masraf gaz ~ HDD + Tolid")
    print("=" * 60)
    model_gaz, clean_gaz, removed_gaz, r2_gaz = fit_with_outlier_removal(
        df, x_cols=["HDD", "tolid"], y_col="gaz", z_thresh=2.5
    )
    print_equation(model_gaz, ["HDD", "Tolid"], "Masraf gaz", r2_gaz)
    if not removed_gaz.empty:
        print("\nNoghat e part e hazf shode:")
        print(removed_gaz.to_string(index=False))
