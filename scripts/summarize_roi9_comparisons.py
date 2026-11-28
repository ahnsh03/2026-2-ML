#!/usr/bin/env python3
"""Verify experiment receipts and produce readable combined ROI9 comparison reports."""
import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quality_estimation.roi9_handoff import sha256
from run_gp_search import save


def verify(directory):
    if (directory/"_SUCCESS").read_text().strip()!=sha256(directory/"artifacts.json"):
        raise ValueError("completion receipt mismatch: "+str(directory))
    for relative,digest in json.loads((directory/"artifacts.json").read_text()).items():
        if sha256(directory/relative)!=digest:
            raise ValueError("artifact hash mismatch: "+relative)


def write_csv(path, rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--gp-run",type=Path,required=True)
    args=p.parse_args()
    root=args.runs
    for directory in [root/"rf",root/"clusters",root/"review",args.gp_run]:
        verify(directory)
    rf=json.loads((root/"rf/summary.json").read_text())["results"]
    gp=json.loads((args.gp_run/"summary.json").read_text())["results"]
    cluster=json.loads((root/"clusters/summary.json").read_text())
    review=json.loads((root/"review/summary.json").read_text())
    rh=json.loads((root/"rf/histories.json").read_text())
    for seed in {h["seed"] for h in rh}:
        arms=[h for h in rh if h["seed"]==seed]
        assert len(arms)==2 and arms[0]["initial_indices"]==arms[1]["initial_indices"]
        assert [t["index"] for t in arms[0]["trials"][:4]]==[t["index"] for t in arms[1]["trials"][:4]]
        assert all(len(h["trials"])==20 and len({t["index"] for t in h["trials"]})==20 for h in arms)
    for k in [8,16,32]:
        for seed in [23411,23412,23413]:
            pair=[r for r in cluster["results"] if r["k"]==k and r["seed"]==seed]
            assert len(pair)==2 and pair[0]["initial_centers_sha256"]==pair[1]["initial_centers_sha256"]
    for r in review["primary_sigma_comparisons"]:
        assert r["added_failures"]-r["removed_failures"]==r["found_delta"]
    assert len(review["results"])==504
    frozen=root/"review/frozen_selection_receipt.json"
    assert json.loads(frozen.read_text())["selections_sha256"]==sha256(root/"review/selections.jsonl")
    out=root/"report"
    out.mkdir(exist_ok=False)
    (out/"figures").mkdir()
    model_rows=[]
    for model,rows in [("RF",rf),("GP",gp)]:
        for method in ["baseline","random","bo"]:
            selected=[r for r in rows if r["method"]==method]
            row=dict(model=model,method=method,repeats=len(selected),cv_mae=float(np.mean([r["cv_mae"] for r in selected])))
            for cohort in ["development","fresh_all","fresh_uncertainty"]:
                key="mae" if model=="RF" else "clipped_mae"
                row[cohort+"_mae"]=float(np.mean([r["cohorts"][cohort][key] for r in selected]))
            model_rows.append(row)
    write_csv(out/"model_comparison.csv",model_rows)
    cluster_rows=[]
    for k in [8,16,32]:
        for method in ["kmeans","emeans"]:
            selected=[r for r in cluster["results"] if r["k"]==k and r["method"]==method]
            row=dict(method=method,k=k,fit_ms=float(np.mean([r["fit_seconds"] for r in selected]))*1000)
            for key in ["mean_squared_distance","same_environment_cluster_agreement","changed_environment_cluster_separation","weather_ami","hour_ami"]:
                row[key]=float(np.mean([r["cohorts"]["development"][key] for r in selected]))
            row["seed_ari"]=[r for r in cluster["stability"] if r["method"]==method and r["k"]==k][0]["development_mean_seed_ari"]
            cluster_rows.append(row)
    write_csv(out/"cluster_comparison.csv",cluster_rows)
    policy_rows=[]
    contrast=[]
    for cohort in ["development","fresh_all","fresh_uncertainty","motion","stream_new","switch","other_road"]:
        for policy in ["random","rf_mean","gp_mean","gp_mean_audit","gp_sigma_audit","gp_sigma_only","gp_lcb","gp_environment"]:
            selected=[r for r in review["results"] if r["cohort"]==cohort and r["policy"]==policy and r["percent"]==10]
            row=dict(cohort=cohort,policy=policy,mode=selected[0]["mode"],budget=selected[0]["actual_budget"],
                     population=selected[0]["population"],total_failures=selected[0]["total_failures"])
            for key in ["found","precision","recall","unique_failure_50m_cells"]:
                row[key]=float(np.mean([r[key] for r in selected]))
            policy_rows.append(row)
        base=[r for r in policy_rows if r["cohort"]==cohort and r["policy"]=="gp_mean_audit"][0]
        sigma=[r for r in policy_rows if r["cohort"]==cohort and r["policy"]=="gp_sigma_audit"][0]
        contrast.append(dict(cohort=cohort,budget=base["budget"],gp_mean_plus_audit_found=base["found"],
                             gp_sigma_plus_audit_found=sigma["found"],found_delta=sigma["found"]-base["found"],
                             mean_failure_cells=base["unique_failure_50m_cells"],sigma_failure_cells=sigma["unique_failure_50m_cells"]))
    write_csv(out/"review_10percent.csv",policy_rows)
    write_csv(out/"sigma_contrast_10percent.csv",contrast)
    save(out/"summary.json",dict(models=model_rows,clusters=cluster_rows,review_10percent=policy_rows,
         sigma_contrast_10percent=contrast,verified_rf_trials=120,verified_cluster_runs=18,verified_policy_arms=504,
         all_results_are_diagnostic=True,threshold_convention="review compares float32 IoU < float32(0.7); equality is normal"))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    rf_rows=[r for r in model_rows if r["model"]=="RF"]
    for ax,key,title in zip(axes,["cv_mae","fresh_all_mae","fresh_uncertainty_mae"],["Training CV","Fresh 480 diagnostic","Extra-road 144 diagnostic"]):
        ax.bar([r["method"] for r in rf_rows],[r[key] for r in rf_rows],color=["gray","orange","royalblue"])
        ax.set(ylabel="MAE (lower is better)",title=title)
    fig.suptitle("ROI9 RF tuning: mean over optimizer seeds")
    fig.tight_layout();fig.savefig(out/"figures/rf_comparison.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for method,color in [("kmeans","orange"),("emeans","royalblue")]:
        rr=[r for r in cluster_rows if r["method"]==method]
        axes[0].plot([r["k"] for r in rr],[r["mean_squared_distance"] for r in rr],"o-",label=method,color=color)
        axes[1].plot([r["k"] for r in rr],[r["same_environment_cluster_agreement"] for r in rr],"o-",label=method,color=color)
    axes[0].set(xlabel="Clusters",ylabel="Development mean squared distance",title="Geometric clustering loss")
    axes[1].set(xlabel="Clusters",ylabel="Same-environment cluster agreement",title="Different-site pairs")
    axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/"figures/cluster_comparison.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    x=np.arange(len(contrast))
    ax.bar(x-.18,[r["gp_mean_plus_audit_found"]/r["budget"] for r in contrast],.36,label="GP mean + 20% audit")
    ax.bar(x+.18,[r["gp_sigma_plus_audit_found"]/r["budget"] for r in contrast],.36,label="GP mean/sigma + 20% audit")
    ax.set_xticks(x,[r["cohort"] for r in contrast],rotation=20)
    ax.set(ylabel="Precision at actual 10% review budget",ylim=(0,1),title="Adding sigma: matched audit and budget, mean over 3 seeds")
    ax.legend();fig.tight_layout();fig.savefig(out/"figures/sigma_comparison.png",dpi=150);plt.close(fig)
    report=["ROI9 후속 비교 결과 · 2026-11-28", "",
        "1. RF 튜닝", "27열·학습 1728장·동일 CV 3분할. 모델 시드 23411과 트리 128개 고정.",
        "현재 설정 / Random Search / BO. 각 탐색 20회(같은 초기 4개 포함), 탐색 시드 3개.",
        "설정은 학습 CV로만 고른 뒤 동결하고 개발·진단 자료를 평가했다."]
    for r in model_rows:
        report.append(f"{r['model']} {r['method']}: CV {r['cv_mae']:.6f}, 개발 {r['development_mae']:.6f}, 새480 {r['fresh_all_mae']:.6f}, 추가144 {r['fresh_uncertainty_mae']:.6f}")
    report += ["RF 튜닝은 CV를 개선했으나 추가 도로의 오차는 커졌다. 이번 결과만으로 모델을 교체하지 않는다.",
        "", "2. e-means와 KMeans", "학습 RGB54에서만 상수 열 제거와 표준화를 적합. 활성 51열.",
        "K=8/16/32 × 3시드 × 2방식 = 18회. 같은 kmeans++ 초기 중심, 최대 5패스, 초기화 1회.",
        "최종 중심의 최근접 배정으로 손실을 계산했다. 같은 패스는 같은 연산량/시간을 뜻하지 않는다."]
    for r in cluster_rows:
        report.append(f"{r['method']} K{r['k']}: 개발 MSE {r['mean_squared_distance']:.4f}, 같은환경·다른위치 일치 {r['same_environment_cluster_agreement']:.2%}, 조건변화 분리 {r['changed_environment_cluster_separation']:.2%}, 시드 ARI {r['seed_ari']:.4f}, CPU 학습 {r['fit_ms']:.2f}ms")
    report += ["K16에서는 KMeans의 거리 손실·동일환경 일치·시드 안정성이 우세했고, e-means는 조건 변화 분리에서 높았다.",
        "K8/32에서는 일부 지표가 반대였다. 군집 수가 다르면 문제 자체가 달라지므로 단일 최적 알고리즘으로 결론 내리지 않는다.",
        "", "3. σ 검수 정책", "수정 전 고정 인계 RF/GP와 고정 e-means K16을 사용해 RF 튜닝 효과와 정책 효과를 분리했다.",
        "정지 풀은 시간순으로 가장하지 않고 전체 풀에서 선택. 주행은 run별 drive_order로 정렬하고 64프레임 창에서 인과적으로 선택했다.",
        "누적 예산 올림의 차이를 창별로 배분해 매 창에서 10%를 별도 올림하는 과다 지출을 피했다.",
        "5/10/20% × 3시드 × 8정책 × 7모집단 = 504회. ID·선택 이유를 GT 평가 전에 파일로 동결했다.",
        "주 비교는 같은 무작위 점검 20%를 둔 GP 평균순과 GP 평균/σ 혼합이다.",
        "혼합은 점검을 뺀 우선 몫의 3/4를 낮은 GP 평균, 나머지를 높은 σ에 배분한다(정수 규칙은 기존 큐와 동일)."]
    for r in contrast:
        report.append(f"{r['cohort']} / 검수 {r['budget']}장: 평균+점검 발견 {r['gp_mean_plus_audit_found']:.2f}, σ혼합+점검 {r['gp_sigma_plus_audit_found']:.2f}, 차이 {r['found_delta']:+.2f}; 실패50m칸 {r['mean_failure_cells']:.2f}→{r['sigma_failure_cells']:.2f}")
    report += ["각 수치는 선택 시드 3개의 평균이다. 소수 발견 수는 한 번의 실제 발견 수가 아니다.",
        "10% 주 비교에서는 7개 모집단 모두 σ 혼합의 발견 수가 감소했다. 일부 주행에서는 위치 다양성이 늘었다.",
        "σ 단독과 GP 평균-σ(LCB)도 별도 평가했다. LCB는 보정된 실패 확률이 아닌 순위 휴리스틱이다.",
        "", "해석 범위", "신규 보정·미사용 최종 평가는 없다. 모든 관측은 이미 조회된 자료의 진단 결과다.",
        "픽셀 중복은 제거하지만 근접·연속 프레임 의존성은 남는다. 3시드는 독립 도로 3개가 아니다.",
        "같은 환경의 이미지 쌍도 독립 표본이 아니므로 쌍 수를 독립 반복 수로 해석하지 않는다.",
        "source site에는 run 접두어가 있어 물리 위치 다양성은 별도의 고정 50m XY 격자로 집계했다. 독립 도로 수를 의미하지 않는다.",
        "검수의 실제 저품질은 저장된 float32 IoU < float32(0.7). 경계와 같은 값은 정상이다.",
        "GP의 과거 float64 변환 지표와 비교하면 개발 경계값 한 행의 분류가 달라질 수 있다. 이번 모든 정책과 차이 집계에는 같은 마스크를 썼다.",
        "review_initial은 경계값 집계 일관성 보완 전 보관본이며, 확정 결과는 review/와 이 report/다.",
        "", "확정 산출물", "model_comparison.csv / cluster_comparison.csv / review_10percent.csv / sigma_contrast_10percent.csv / figures/",
        "모든 예산의 원자료: ../review/summary.csv. 선택 ID와 사유: ../review/selections.jsonl.",
        "GP 비교의 원자료는 summary.json에 연결한 gp_run 경로를 사용한다."]
    (out/"REPORT.txt").write_text("\n".join(report)+"\n")
    save(out/"verification.json",dict(rf_trials=120,cluster_runs=18,policy_arms=504,shared_initials_verified=True,
          sigma_added_minus_removed_verified=True,input_artifact_hashes_verified=True,gp_run=str(args.gp_run),
          stages={stage:sha256(root/stage/"artifacts.json") for stage in ("rf","clusters","review")}))
    (out/"command.txt").write_text("PYTHONPATH=src "+shlex.join([sys.executable]+sys.argv)+"\n")
    save(out/"artifacts.json",{str(f.relative_to(out)):sha256(f) for f in out.rglob("*") if f.is_file()})
    (out/"_SUCCESS").write_text(sha256(out/"artifacts.json")+"\n")
    print("Verified and summarized:",out)


if __name__=="__main__":
    main()
