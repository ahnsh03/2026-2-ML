#!/usr/bin/env python3
"""Publish local handoff evidence for the exact NAS r2 48/590 reproduction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from compare_official_review_replay import read_completed,sha
from quality_estimation.review_targets import support_aware_targets


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def quality_metrics(prediction,truth,groups):
    predicted=np.array([r["estimated_iou"]for r in prediction])
    defined=np.isfinite(truth)
    by_group={str(g):float(np.mean(abs(predicted[(groups==g)&defined]-truth[(groups==g)&defined])))
              for g in np.unique(groups) if np.any((groups==g)&defined)}
    return dict(defined_rows=int(defined.sum()),frame_mae=float(np.mean(abs(predicted[defined]-truth[defined]))),
                segment_macro_mae=float(np.mean(list(by_group.values()))),mae_by_segment=by_group)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args();root=args.root
    out=root/"report";out.mkdir(exist_ok=False);(out/"source").mkdir()
    release=read(root/"release_verification.json")
    assert release["status"]=="verified" and release["rgb_examples"]==48
    raw=read(root/"raw_copy_receipts.json")
    assert len(raw)==1180 and len({(r["sample_id"],r["kind"])for r in raw})==1180
    for row in raw:
        path=root/("data_rgb"if row["kind"]=="image"else"data_gt")/row["relative"]
        assert sha(path)==row["sha256"],row["sample_id"]
    s48=read(root/"standalone48_comparison.json");s590=read(root/"replay590_comparison.json")
    assert s48["status"]=="matched" and s48["selection_overlap"]==24
    assert s590["selection_ids_order_reasons_equal"] and s590["selection_overlap"]==59
    assert s590["failures"]==["features.X"],s590["failures"]
    _,gp,candidates=read_completed(root/"replay590_primary/gpu")
    _,rf,rf_candidates=read_completed(root/"replay590_rf/gpu")
    rows=read(root/"nas_metadata/confirmation/registry/index_frozen.json")["confirmation"]["rows"]
    ids=[r["sample_id"]for r in rows]
    assert ids==[r["sample_id"]for r in gp]==[r["sample_id"]for r in rf]
    frozen=read(root/"rgb_selection_frozen.json")
    assert frozen["primary_candidates_sha256"]==sha(root/"replay590_primary/gpu/candidates.jsonl")
    assert frozen["rf_candidates_sha256"]==sha(root/"replay590_rf/gpu/candidates.jsonl")
    recompute=read(root/"gt590_recomputed/results/run.json")
    assert recompute["count_exact_rows"]==590 and recompute["max_count_difference"]==0
    gt_join=records(root/"gt590_join/gt_review.jsonl")
    original_join=records(root/"nas_metadata/confirmation/runtime/rtx50-cu128_gt/gt_review.jsonl")
    assert [r["sample_id"]for r in gt_join]==[r["sample_id"]for r in original_join]
    for a,b in zip(gt_join,original_join):
        for field in ["true_lane_iou","near_gt_pixels","far_gt_pixels","supported_error_targets"]:
            assert a[field]==b[field],field
    with np.load(root/"gt590_recomputed/results/recomputed_counts.npz")as d:
        assert d["ids"].tolist()==ids
        truth=d["y"].copy();counts=d["counts"].copy()
    targets=support_aware_targets(counts)
    groups=np.array([r["group"]for r in rows])
    actual={"primary":quality_metrics(gp,truth,groups),"RF_only":quality_metrics(rf,truth,groups)}
    reference=read(root/"nas_metadata/confirmation/review_assessment/summary.json")["results"]
    metric_rows=[]
    for name in actual:
        ref=reference[name]
        metric_rows.append(dict(model="GP52"if name=="primary"else"RF52",
            reference_frame_mae=ref["quality_mae"],actual_frame_mae=actual[name]["frame_mae"],
            reference_segment_macro_mae=ref["quality_segment_macro_mae"],
            actual_segment_macro_mae=actual[name]["segment_macro_mae"],
            absolute_macro_difference=abs(actual[name]["segment_macro_mae"]-ref["quality_segment_macro_mae"])))
        assert metric_rows[-1]["absolute_macro_difference"]<1e-4
    with (out/"quality_comparison.csv").open("w",newline="")as stream:
        w=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    selected={r["sample_id"]for r in candidates}
    chosen=np.array([sid in selected for sid in ids])
    errors=targets["near"]["error"]
    positives=int(errors.sum());found=int((errors&chosen).sum())
    assert positives==107 and found==53 and int(chosen.sum())==59
    hours=np.array([r["hour"]for r in rows])
    by_hour={str(h):dict(rows=int((hours==h).sum()),near_errors=int((errors&(hours==h)).sum()),
                selected_near_errors=int((errors&chosen&(hours==h)).sum()))for h in np.unique(hours)}
    review=dict(population=590,selected=59,near_errors=positives,selected_near_errors=found,
                recall=found/positives,precision=found/59,by_hour=by_hour)
    np.testing.assert_allclose(review["recall"],reference["primary"]["review"]["near_error_recall"],atol=1e-12,rtol=0)
    with np.load(root/"nas_metadata/confirmation/review_assessment/RF_only_predictions.npz")as z:
        assert z["ids"].tolist()==ids
        rf_reference_delta=float(np.max(abs(z["estimated_iou"]-np.array([r["estimated_iou"]for r in rf]))))
    original_ids=reference["RF_only"]["selected_ids"]
    rf_ids=[r["sample_id"]for r in rf_candidates]
    diagnostic=read(root/"feature_diagnostic_plan.json")
    with np.load(root/"feature_diagnostic/gpu/features.npz")as a,np.load(root/"replay590_primary/gpu/features.npz")as b:
        start=diagnostic["original_batch_start"];n=diagnostic["batch_rows"]
        assert a["ids"].tolist()==b["ids"][start:start+n].tolist()
        repeat_delta=float(np.max(abs(a["X"]-b["X"][start:start+n])))
    diagnostic["same_gpu_original_batch_repeat_max_feature_difference"]=repeat_delta
    diagnostic["interpretation"]="One cross-machine reference deviation; exact original batch repeats on this GPU. Cause not isolated; hardware/numerical effects possible. No policy/model changes."
    summary=dict(status="completed_with_one_reported_feature_tolerance_deviation",date_kst="2026-11-29",
        scope="Exact official NAS r2 stored RGB replay and post-selection GT reproduction; no ROS",
        all_five_steps_performed=True,official48=s48,official590=s590,
        raw_file_hashes_verified=1180,release_file_hashes_verified=release["files"],
        quality=actual,quality_reference_comparison=metric_rows,review=review,
        independent_raw_gt_recomputation=recompute,selected_gt_records_exact_match=True,
        rf_reference_max_q_difference=rf_reference_delta,
        rf_reference_selected_ids_order_equal=(rf_ids==original_ids),
        feature_deviation=diagnostic,models_or_thresholds_changed=False,
        gt_available_to_rgb_inference_process=False,live_ros_executed=False,actual_rtx40_gpu_tested=False,
        nas_modified=False,external_message_or_pr_sent=False)
    save(out/"summary.json",summary)
    save(out/"review_comparison.json",review)
    save(out/"feature_deviation.json",diagnostic)
    report=["장원태 · 공식 r2 자료 독립 재현 인계 보고서 · 2026-11-29", "",
        "결론: 요청한 1~5단계를 수행했다. 공식 48/590 입력·선택·GT 성능은 재현됐고, 특징 한 값의 수치 비교 기준 미달은 별도 보존했다.",
        "", "1. 자료 복사·검증", "NAS 원본은 읽기만 했다. 로컬에 별도로 복사했으며 원본은 변경하지 않았다.",
        "원본: \\\\Aim-nas\\aim_2026\\개인 폴더\\안승현\\ML\\ml_scene_cycle_20261128",
        f"48장 release manifest의 {release['files']}개 파일 해시 검증. RGB590+GT590 파일 총1180개를 공식 기대 SHA-256과 대조.",
        "48장 배포 묶음, 590장 입력/GT 목록·source index·pack/schema·기준 출력·지표를 보존했다.",
        "590장 RGB·GT는 분리 저장했다. RGB 추론 컨테이너에는 GT 저장소를 마운트하지 않았다.",
        "", "2. 공식48장 standalone", "원래 code commit 287b1b5·r2 가중치·GP52/RF74 번들·batch4·budget24 유지.",
        "실제 RTX5070/CUDA12.8에서 배포 묶음과 새 출력 폴더만 마운트해 실행했다. 원래 작업공간/원본 볼륨은 마운트하지 않았다.",
        f"선택24장 ID/순서/사유 및 환경 군집 일치. Q 최대 차이 {s48['max_numeric_differences']['estimated_iou']:.10g}.",
        f"특징 최대 차이 {s48['feature_differences']['X']:.10g}; 비교 기준1e-4 이내.",
        "", "3. 공식590장 GT 없는 재생·동결", "원래 RGB manifest 바이트·순서와 모델/정책을 유지했다. batch4·budget59, 원래 무작위 감사5개 유지.",
        "Primary(GP52/RF74)와 RF_only(RF52)를 각각 실행하고, 선택 파일 SHA를 저장한 다음 GT 평가를 시작했다.",
        f"Primary 선택59장 ID/순서/사유와 군집은 원래 GPU 기준과 일치. Q 최대 차이 {s590['max_numeric_differences']['estimated_iou']:.10g}.",
        f"RF52의 공식 prediction NPZ 대비 Q 최대 차이 {rf_reference_delta:.10g}. RF_only 선택 순서 일치: {rf_ids==original_ids}.",
        "", "4. GT 연결·성능 재계산", "공식 join_spatial_gt로 선택59장의 원본바이트/픽셀/ROI/인지 버전을 검사해 연결했고, 기준 GT 레코드와 일치했다.",
        "추가로 원본 Semantic을 새로 디코딩하고 r2를 재추론해590장의 ROI별 TP/FP/FN을 독립 재계산했다. 공식 pack과590/590장 정확히 일치했다.",
        "README의0.03136/0.02421은 단순 프레임 평균이 아니라 구간별 MAE의 동일 가중 평균이다. 아래에 둘 다 기록한다."]
    for r in metric_rows:
        report.append(f"{r['model']}: 기준 구간평균 MAE {r['reference_segment_macro_mae']:.9f} / 재현 {r['actual_segment_macro_mae']:.9f}; 기준 프레임평균 {r['reference_frame_mae']:.9f} / 재현 {r['actual_frame_mae']:.9f}")
    report += [f"근거리 오류107장 중53장 회수: recall {review['recall']:.9f}, 검수59장 precision {review['precision']:.9f}.",
        "시간대 분모·회수 수: "+json.dumps(by_hour,ensure_ascii=False),
        "", "수치 차이: 숨기지 않고 보존",f"{diagnostic['sample_id']} / {diagnostic['feature']}: 기준과 차이 {diagnostic['observed_max_difference']:.10g}.",
        "590×475 특징 중 한 값만1e-4를 넘었다. strict comparison 상태는 differences_detected로 남겼다.",
        f"같은 GPU·같은 원래4장 배치의 특징 반복 차이는 {repeat_delta:.10g}였다. 환경 군집·검수 순위·오류 점수·GT counts에는 영향이 없었다.",
        "GPU/수치 환경 차이가 가능한 원인이지만 요인 분리 실험은 하지 않았다. 원인을 확정했다고 쓰지 않는다.",
        "", "5. 인계", "summary.json, quality_comparison.csv, review_comparison.json, feature_deviation.json이 핵심 결과다.",
        "source/와 REPRODUCE.sh에 검증/GT 재계산 코드와 재생 명령을 보존했다. 원본/모델/정책/기준 출력은 고정했으며 재튜닝하지 않았다.",
        "실제 ROS, 실제40GPU, 새 도시 일반화, 새 모델 학습·채택을 검증한 결과가 아니다.",
        "이 문서는 로컬 인계 준비물이다. 팀원에게 메시지를 전송하거나 PR을 게시하지 않았다."]
    (out/"TEAM_HANDOFF.txt").write_text("\n".join(report)+"\n")
    for path in [Path(__file__),Path("scripts/compare_official_review_replay.py"),Path("scripts/recompute_r2_confirmation_gt.py"),
                 Path("src/quality_estimation/review_targets.py"),Path("src/quality_estimation/join_spatial_gt.py")]:
        shutil.copy2(path,out/"source"/path.name)
    save(out/"source_sha256.json",{p.name:sha(p)for p in (out/"source").iterdir()})
    print(json.dumps(dict(status=summary["status"],official48_selected=24,official590_selected=59,
                         near_error_found=found,near_error_total=positives,quality=metric_rows,feature_repeat_delta=repeat_delta),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
