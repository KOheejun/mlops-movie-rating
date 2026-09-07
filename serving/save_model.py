"""
save_model.py

기능
- 최신 체크포인트(.pkl) 자동 탐색
- MoviePredictor 모델 복원
- BentoML Model Store에 저장 (자동 버전 생성)
- bentoml_registry.json으로 STAGING / PROD 포인터 관리

사용법
1) STAGING 저장
   python serving/save_model.py save --stage staging

2) STAGING → PROD 승격
   python serving/save_model.py promote --latest-staging

3) 레지스트리 확인
   python serving/save_model.py show
"""

import os
import sys
import glob
import json
import pickle
import argparse
from datetime import datetime

# ======================================================
# 경로 설정 (serving/save_model.py 기준, 안전한 절대경로)
# ------------------------------------------------------
# 이 리포에서는 학습 코드가 <repo_root>/src/ 에 있으므로
# (원본 프로젝트에서는 <repo_root>/mlops/src/ 였음),
# repo 루트를 sys.path에 추가해 `from src...` import가 되게 한다.
# ======================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import bentoml
from src.utils.utils import model_dir
from src.model.movie_predictor import MoviePredictor

# ======================================================
# 레지스트리 파일 (serving 폴더 안)
# ======================================================
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "bentoml_registry.json")


def _utc_now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {
            "model_name": "movie_rating_model",
            "staging_tag": None,
            "prod_tag": None,
            "history": []
        }
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(reg):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


# ======================================================
# 체크포인트 로드
# ======================================================
def load_latest_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    pattern = os.path.join(target_dir, "*.pkl")
    model_files = glob.glob(pattern)

    if not model_files:
        raise FileNotFoundError(f"No checkpoint (.pkl) found in {target_dir}")

    latest_ckpt = max(model_files, key=os.path.getmtime)
    print(f"[CHECKPOINT] Loading: {latest_ckpt}")

    with open(latest_ckpt, "rb") as f:
        checkpoint = pickle.load(f)

    for k in ["model_params", "model_state_dict"]:
        if k not in checkpoint:
            raise KeyError(f"Checkpoint missing key: {k}")

    return checkpoint, latest_ckpt


# ======================================================
# BentoML 저장
# ======================================================
def save_to_bentoml(stage="staging"):
    checkpoint, ckpt_path = load_latest_checkpoint()

    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # MoviePredictor가 torch.nn.Module이 아닐 수도 있어서 방어
    if hasattr(model, "eval"):
        model.eval()

    scaler = checkpoint.get("scaler")
    scaler_y = checkpoint.get("scaler_y")

    bento_model = bentoml.picklable_model.save_model(
        "movie_rating_model",
        model,
        custom_objects={
            "scaler": scaler,
            "scaler_y": scaler_y
        },
        metadata={
            "stage": stage,
            "saved_at_utc": _utc_now(),
            "checkpoint_path": ckpt_path,
            "model_params": checkpoint["model_params"]
        }
    )

    tag = str(bento_model.tag)
    print(f"[BENTOML] Saved: {tag}")

    reg = _read_registry()
    reg["model_name"] = "movie_rating_model"

    if stage == "staging":
        reg["staging_tag"] = tag
    elif stage == "prod":
        reg["prod_tag"] = tag

    reg["history"].append({
        "tag": tag,
        "stage": stage,
        "saved_at_utc": _utc_now(),
        "checkpoint_path": ckpt_path
    })

    _write_registry(reg)
    print(f"[REGISTRY] Updated: {REGISTRY_PATH}")

    return tag
    

# ======================================================
# PROD 승격
# ======================================================
def promote_to_prod(tag):
    # BentoML에 실제 존재하는지 확인
    bentoml.models.get(tag)

    reg = _read_registry()
    reg["prod_tag"] = tag
    reg["prod_promoted_at_utc"] = _utc_now()
    _write_registry(reg)

    print(f"[PROMOTE] PROD <- {tag}")


def promote_latest_staging():
    reg = _read_registry()
    tag = reg.get("staging_tag")
    if not tag:
        raise RuntimeError("No staging_tag found. Save a staging model first.")
    promote_to_prod(tag)


def show_registry():
    reg = _read_registry()
    print(json.dumps(reg, ensure_ascii=False, indent=2))


# ======================================================
# CLI
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save")
    p_save.add_argument("--stage", default="staging", choices=["staging", "prod"])

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--tag")
    p_promote.add_argument("--latest-staging", action="store_true")

    sub.add_parser("show")

    args = parser.parse_args()

    if args.cmd == "save":
        save_to_bentoml(stage=args.stage)

    elif args.cmd == "promote":
        if args.latest_staging:
            promote_latest_staging()
        else:
            if not args.tag:
                raise ValueError("Use --tag or --latest-staging")
            promote_to_prod(args.tag)

    elif args.cmd == "show":
        show_registry()


if __name__ == "__main__":
    main()