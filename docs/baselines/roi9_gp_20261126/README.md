# 9영역28열·공간GP·26R1 공개 근거

11월25–26일의 집계와 그림이다. 결론·자료 역할·미완료 범위는 [실험 보고](../../experiments/ROI9_GP와_26R1_검증_20261126.md), 현행 절차는 [가이드](../../guides/ROI와_특징추출.md), 순서 있는28열·모델·보정 해시는 [v3 명세](../../../config/region_feature_recipe_v3.json)를 따른다. 원본·상세ID/경로·학습 구성원·모델 파일은 실험 볼륨에서 인계한다.

## 자료와 비교표

| 파일 | 내용과 읽을 때의 범위 |
|---|---|
| [data_inventory.json](data_inventory.json) | 82팩23,142장·원본RGB픽셀/ID 중복·팩해시·시뮬레이터 출처. 신규26R1은336장 |
| [feature_screen.json](feature_screen.json), [feature_pruning.json](feature_pruning.json) | 기존1,728장 단계의 후보 비교·축소. 기존24열에 조건부인 CV와 개발 선택이며 완전 중첩 선택 검증이 아님 |
| [coverage_and_coupling.json](coverage_and_coupling.json) | 공통2,480장 확대학습·지리fold·정답 미정의 제외. 이미 진단한624장을 학습에 승격. GP36/RF12 튜닝 설정으로 횟수는 다름 |
| [archival_evaluation.json](archival_evaluation.json) | 보관반복주행/별도링크의 진단. 확대학습 전 이미 보았으므로 미사용 새 평가라고 부르지 않음 |
| [test26_transfer.json](test26_transfer.json) | 6기존품질학습위치×24조건=144장의 설정 전이. 새지리 아님 |
| [test26_heldroads.json](test26_heldroads.json) | 모델/보정 고정 뒤4품질학습미사용링크96장. 같은지도이며4곳의 한계가 있음 |
| [paired_group_differences.json](paired_group_differences.json) | 프레임이 아닌 세션/위치 단위 GP−RF 짝 MAE차·bootstrap구간 |
| [condition_control.json](condition_control.json) | GT날씨/시각만 아는 중앙값 대조와 조건 안의 순위 성적 |
| [sigma_hypothesis.json](sigma_hypothesis.json) | 큰σ→전체저하/작은σ→원거리약화 가설의 정확한 부호·사건·AUC. 가설 미채택 |
| [review_tradeoff.json](review_tradeoff.json) | 64프레임 창·올림10% 실제805슬롯의 절충. 원거리약화는 운행실패 정답이 아님 |
| [label_acquisition.json](label_acquisition.json) | 이미 진단한624장 풀에서192정답 선택의 후향 재생. 신규온라인active learning 아님 |
| [additive_bo_random.json](additive_bo_random.json) | 가산커널·EI BO와무작위 각각18회. 이번 예산에서BO이득 미확인 |
| [spatial_target_support.json](spatial_target_support.json), [13 정답분포](13_far_target_support.png) | 새 보정의 원거리 recall≥.5 샘플0장 대 평가25장. 모델을 바꾸지 않은 사후 진단 |
| [final_data_audit.json](final_data_audit.json) | 82팩 해시·전체 정답/TPFPFN 일치·ID/픽셀 중복·신규 RGB/Semantic672파일 재확인 |
| [runtime_verification.json](runtime_verification.json) | RTX50실행·40CPU대조·이전14,955장 변동·GT없는재생/사후GT연결 범위 |
| [bundle_receipt.json](bundle_receipt.json) | Git밖 동결번들의 파일해시와 정확학습구성원에서X/가중치 재구성 영수증 |
| [publication_receipt.json](publication_receipt.json) | 공개한 파일/그림의SHA-256 |

[protocols](protocols/)에는 후보 스키마, 학습/개발/보정 키, 특징제거 구성·선정28열을 둔다. `feature_screen.json`은 원래 계획의 nested 문구를 실제 수행 범위로 정정한 **사후 설명본**이며 원래 계획 해시를 남겼다. 외부 평가·시각화에서 다시 고른 설정을 미사용검증으로 설명하지 않는다.

## 발표·페이지용 그림

| 그림 | 설명과 제한 |
|---|---|
| [01 ROI/선정열](01_roi9_selected_inputs.png) | 고정9영역·직접 입력수. 원거리는관계,상부는전체통계에간접기여 |
| [02 특징 비교](02_feature_screen.png) | 최초학습단계의비교. 독립성적과구분 |
| [03 σ와보조평균](03_sigma_and_spatial_means.png) | 동일far-weak부분집합에서far-only를양성으로한 **+σ** AUC. 원가설의−σ 방향은sigma_hypothesis.json참조 |
| [04 학습범위](04_gp_rf_coverage.png) | 입력28열고정·품질학습자료확대. 인지모델f0는동일 |
| [05 선별절충](05_review_tradeoff.png) | 전체저품질발견과별도공간진단의절충 |
| [06 기존보정](06_calibration_coverage_width.png) | 기존576장보정의이력. 새26R1보정그림은11번 |
| [08 오프라인흐름](08_offline_pipeline.png) | RGB만으로점수/큐를만든뒤GT검토·인지재학습으로연결 |
| [09 자료역할별GP/RF](09_setup_transfer.png) | 자료역할과도로별로순위가다름 |
| [10 근원후분포](10_spatial_posterior.png) | 유형별첫유효사례.95%Gaussian타원은보장된공동포함률아님 |
| [11 원거리보정한계](11_far_recall_reliability.png) | 새링크의Q/근/원90%포함률91.7/85.4/39.6%.공간점수는진단용 |
| [12 낮1](12_held_case_01.png), [밤1](12_held_case_02.png), [낮2](12_held_case_05.png), [밤2](12_held_case_06.png) | 처음두새링크의고정Sunny13/20사례. 오차크기로선정하지않음 |

12번은 원래RGB·짝Semantic 차선GT·정확한 재생출력·TP/FP/FN을 정합했다. GT는640×360 최근접변환,softmax는12+12패딩을제외했다. PNG는원래종횡비를유지해사용한다. E실험볼륨에는SVG와16개사례의원본16:9구성요소도별도로보존했다. 기존07번은학습/진단사례여서현재공개예시에서는새링크12번을사용한다.

실제40GPU·라이브ROS는이번검증범위밖이다. 현재28열/GP를이용한새인지재학습과최종정책채택도아직남아있다.
