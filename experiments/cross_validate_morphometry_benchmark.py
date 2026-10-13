#!/usr/bin/env python3
"""Morphometry benchmark on the exact PersLay grouped outer folds.

Expected morphometry CSV: one row per sample with columns
    structure_id (or relative_image/filename), variant, <numeric features...>
For example:
    structure_0000,baseline,0.31,0.28,0.54,...
    structure_0000,uniform,0.25,0.23,0.61,...
    structure_0000,weakened,0.25,0.27,0.58,...

The target and BV/TV are read from the FEA CSVs. The split assignments are read
from the PersLay CV output, so every benchmark uses the identical outer test
structures. Ridge and gradient-boosting regressors are provided. Hyperparameters
are selected on that fold's existing inner validation subset only.
"""
from __future__ import annotations
import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sid(value: str) -> str:
    m = re.search(r"(structure_\d+)", str(value), flags=re.I)
    if not m: raise ValueError(f"Cannot extract structure ID from {value!r}")
    return m.group(1).lower()


def read_fea(data_root: Path, variants, target_col, bvtv_col):
    rows=[]
    for v in variants:
        p=data_root/f"fea_results_{v}_ds3.csv"
        d=pd.read_csv(p)
        d=d[(d["status"].astype(str).str.lower()=="success") & (d["converged"].astype(str).str.lower().isin(["true","1","yes"]))].copy()
        d["structure_id"]=d["relative_image"].map(sid); d["variant"]=v; d["sample_id"]=v+"__"+d["structure_id"]
        rows.append(d[["sample_id","structure_id","variant",target_col,bvtv_col]])
    return pd.concat(rows,ignore_index=True)


def metrics(y,p):
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(mean_squared_error(y,p)**0.5),"r2":float(r2_score(y,p))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("data")); ap.add_argument("--cv-output",type=Path,required=True)
    ap.add_argument("--morphometry-csv",type=Path,required=True); ap.add_argument("--output-dir",type=Path,default=None)
    ap.add_argument("--variants",nargs="+",default=["baseline","uniform","weakened"])
    ap.add_argument("--target-column",default="apparent_youngs_modulus_MPa"); ap.add_argument("--bvtv-column",default="retained_density_bvtv")
    ap.add_argument("--models",nargs="+",choices=["ridge","gbr"],default=["ridge","gbr"])
    ap.add_argument("--include-bvtv",action="store_true",help="Include BV/TV alongside morphometry (recommended main benchmark).")
    ap.add_argument("--feature-columns",nargs="+",default=None,help="Explicit numeric morphometry columns. Otherwise all numeric non-ID columns are used.")
    args=ap.parse_args(); out=args.output_dir or (args.cv_output/"morphometry_benchmark"); out.mkdir(parents=True,exist_ok=True)
    fea=read_fea(args.data_root,args.variants,args.target_column,args.bvtv_column)
    morph=pd.read_csv(args.morphometry_csv)
    if "structure_id" not in morph:
        source="relative_image" if "relative_image" in morph else "filename" if "filename" in morph else None
        if source is None: raise SystemExit("Morphometry CSV needs structure_id, relative_image, or filename.")
        morph["structure_id"]=morph[source].map(sid)
    else: morph["structure_id"]=morph["structure_id"].map(sid)
    if "variant" not in morph: raise SystemExit("Morphometry CSV needs a variant column: baseline/uniform/weakened.")
    morph["variant"]=morph["variant"].astype(str).str.lower(); morph["sample_id"]=morph["variant"]+"__"+morph["structure_id"]
    data=fea.merge(morph,on=["sample_id","structure_id","variant"],how="inner",validate="one_to_one")
    if len(data)!=len(fea):
        missing=sorted(set(fea.sample_id)-set(data.sample_id)); raise SystemExit(f"Missing morphometry for {len(missing)} FEA samples, e.g. {missing[:10]}")
    excluded={"sample_id","structure_id","variant",args.target_column,args.bvtv_column,"relative_image","filename","status","converged"}
    if args.feature_columns:
        features=list(args.feature_columns)
    else:
        features=[c for c in data.columns if c not in excluded and pd.api.types.is_numeric_dtype(data[c])]
    if args.include_bvtv and args.bvtv_column not in features: features=[args.bvtv_column]+features
    if not features: raise SystemExit("No numeric morphometry features selected.")
    assignments=pd.read_csv(args.cv_output/"fold_assignments.csv")
    assignments=assignments.drop_duplicates(["repeat","outer_fold","sample_id","split"])
    rows=[]; preds=[]
    repeats=sorted(assignments.repeat.unique()); folds=sorted(assignments.outer_fold.unique())
    for rep in repeats:
      for fold in folds:
        a=assignments[(assignments.repeat==rep)&(assignments.outer_fold==fold)]
        split_map=dict(zip(a.sample_id,a.split))
        d=data[data.sample_id.isin(split_map)].copy(); d["split"]=d.sample_id.map(split_map)
        tr=d[d.split=="train"]; va=d[d.split=="validation"]; te=d[d.split=="test"]
        Xtr=tr[features]; ytr=tr[args.target_column].to_numpy(float); Xva=va[features]; yva=va[args.target_column].to_numpy(float); Xte=te[features]; yte=te[args.target_column].to_numpy(float)
        for model_name in args.models:
          if model_name=="ridge":
            candidates=[0.0,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0]
            best=None
            for alpha in candidates:
              pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
              pipe.fit(Xtr,ytr); p=pipe.predict(Xva); score=mean_squared_error(yva,p)**0.5
              if best is None or score<best[0]: best=(score,alpha)
            alpha=best[1]
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=alpha))])
            param=f"alpha={alpha}"
          else:
            candidates=[(1,0.03),(1,0.05),(2,0.03),(2,0.05),(2,0.1),(3,0.03),(3,0.05)]
            best=None
            for depth,lr in candidates:
              model0=Pipeline([("impute",SimpleImputer(strategy="median")),("model",GradientBoostingRegressor(n_estimators=300,max_depth=depth,learning_rate=lr,loss="squared_error",random_state=2026+int(rep)*100+int(fold)))])
              model0.fit(Xtr,ytr); p=model0.predict(Xva); score=mean_squared_error(yva,p)**0.5
              if best is None or score<best[0]: best=(score,depth,lr)
            _,depth,lr=best
            model=Pipeline([("impute",SimpleImputer(strategy="median")),("model",GradientBoostingRegressor(n_estimators=300,max_depth=depth,learning_rate=lr,loss="squared_error",random_state=2026+int(rep)*100+int(fold)))])
            param=f"depth={depth},lr={lr}"
          # Refit on train+validation after choosing hyperparameter; outer test stays untouched.
          dev=pd.concat([tr,va]); model.fit(dev[features],dev[args.target_column]); p=model.predict(Xte)
          for scope in ["all"]+sorted(te.variant.unique()):
            mask=np.ones(len(te),dtype=bool) if scope=="all" else (te.variant.to_numpy()==scope)
            m=metrics(yte[mask],p[mask]); rows.append({"repeat":rep,"outer_fold":fold,"model":model_name,"variant_scope":scope,"hyperparameter":param,"n":int(mask.sum()),**m})
          for sample_id,structure_id,variant,t,pp in zip(te.sample_id,te.structure_id,te.variant,yte,p): preds.append({"repeat":rep,"outer_fold":fold,"model":model_name,"sample_id":sample_id,"structure_id":structure_id,"variant":variant,"target_MPa":t,"prediction_MPa":float(pp)})
    pd.DataFrame(rows).to_csv(out/"fold_metrics.csv",index=False); pd.DataFrame(preds).to_csv(out/"predictions.csv",index=False)
    summ=[]
    for key,g in pd.DataFrame(rows).groupby(["model","variant_scope"]):
        summ.append({"model":key[0],"variant_scope":key[1],"rmse_mean":g.rmse.mean(),"rmse_std":g.rmse.std(ddof=1),"r2_mean":g.r2.mean(),"mae_mean":g.mae.mean()})
    pd.DataFrame(summ).to_csv(out/"metric_summary.csv",index=False)
    (out/"config.json").write_text(json.dumps({"features":features,"include_bvtv":args.include_bvtv,"variants":args.variants},indent=2)+"\n")
    print(pd.DataFrame(summ).to_string(index=False)); print(f"\nFeatures ({len(features)}): {features}")

if __name__=="__main__": main()
