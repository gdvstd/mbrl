# Gilbreth migration — VLA transfer track

로컬(M1)에서 검증 완료된 것: LIBERO 파이프라인 전체(관측 정합·replay·평가 프로토콜),
baseline 4종(finetune/aa/jsrl/ksrl) 학습 루프, task split(seen 0-4 / unseen 5-9).
로컬에서 막힌 것: teacher 품질 (소박한 BC는 compounding error로 0%) → 여기서는
진짜 VLA를 teacher로 파인튜닝한다.

## 순서

1. **코드 업로드** (로컬에서):
   ```bash
   rsync -av --exclude venv --exclude LIBERO --exclude runs_vla \
       ~/Desktop/Purdue/CS49000/MBRL/vla/ \
       nam120@gilbreth.rcac.purdue.edu:'$CLUSTER_SCRATCH/vla/code/'
   ```
2. **환경 구축** (login node): `bash code/cluster/setup_gilbreth.sh`
3. **렌더링 스모크** (GPU node — 반드시 먼저):
   `srun -A joecamp --gpus=1 -t 0:10:00 python $CLUSTER_SCRATCH/vla/smoke_render.py`
   실패 시 MUJOCO_GL=osmesa 로 재시도 (느리지만 CPU로 동작).
4. **데모 다운로드**: `cd $CLUSTER_SCRATCH/vla && bash code/cluster/fetch_all_demos.sh`
5. **Teacher: OpenVLA-7B LoRA 파인튜닝 (seen 5 task)** — 별도 안내 예정.
   openvla repo의 LIBERO 파인튜닝 레시피 사용; seen-5 필터링한 RLDS 데이터 필요.
   A100-40GB 1장, LoRA r=32 기준 ~27GB. 모델 선택 근거: 토큰 기반 action head라
   이후 EBTL energy(φ=logsumexp(logits))가 그대로 성립 (SmolVLA는 flow-matching이라 불가).
6. **Teacher 게이트 평가**: seen ≫ unseen 확인 후에만 baseline 비교 진행.
7. **Baselines**: `sbatch code/cluster/sbatch_student.sh <transfer> <seed>`
   (transfer ∈ scratch|finetune|aa|jsrl|ksrl × seed 0-2; A100 9장이면 전부 동시 가능)

## 주의 (로컬에서 밟은 지뢰들, setup 스크립트에 반영됨)

- mujoco는 **2.3.7 고정** (3.x는 robosuite 1.4.1과 비호환)
- LIBERO는 pip wheel이 비어 있음 → clone + .pth 방식
- torch≥2.6의 weights_only 기본값이 LIBERO init 파일 로드를 깨뜨림
  → vla_common.get_init_states가 우회 처리 (코드에 반영됨)
- 클러스터 headless 렌더링: `MUJOCO_GL=egl` (GPU 노드에서만)

## 남은 작업 (코드 측)

- [ ] OpenVLA teacher용 FrozenTeacher 어댑터 (HF 모델 로드, 토큰 action 디코딩,
      distribution/log-prob 인터페이스를 vla_ppo_common.FrozenBCTeacher와 동일하게)
- [ ] seen-5 RLDS 필터 스크립트 (OpenVLA 파인튜닝 입력용)
- [ ] (EBTL 단계) energy_score: action-token raw logits logsumexp
