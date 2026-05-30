# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence


TEXT_WEIGHT = 0.8
VOICE_WEIGHT = 0.2
SMOOTHING_PREVIOUS_WEIGHT = 0.65
ALERT_THRESHOLD = 70.0
DEFAULT_SCORING_MODE = os.environ.get("RISK_SCORING_MODE", "gated_v1").strip() or "gated_v1"

FRAUD_ACTION_TERMS = [
    "验证码",
    "验证马",
    "短信验证",
    "安全账户",
    "账户冻结",
    "冻结账户",
    "转账",
    "汇款",
    "屏幕共享",
    "远程控制",
    "公检法",
    "公安局",
    "检察院",
    "法院",
    "通缉",
    "洗钱",
    "涉案",
    "刷流水",
    "银行卡",
    "核验账户",
    "核验",
    "屏幕",
    "共享",
    "下载会议",
    "会议协助",
    "远程协助",
]

URGENCY_TERMS = [
    "马上",
    "立即",
    "现在",
    "尽快",
    "否则",
    "逾期",
    "不要挂",
    "保密",
    "不要退出",
    "不要关闭",
    "不要挂断",
    "一分钟",
]

FRAUD_CONTEXT_TERMS = [
    "异常交易",
    "异常登录",
    "异常资金",
    "异地登录",
    "风险处理",
    "安全验证",
    "身份核验",
    "人工核验",
    "资金来源",
    "临时限制",
    "账户安全",
    "支付账户",
    "退款通道",
    "系统验证",
    "风险拦截",
    "本人操作",
    "验证流程",
]

BUSINESS_FINANCE_TERMS = [
    "工资",
    "社保",
    "五险一金",
    "合同",
    "续签",
    "签约",
    "费用",
    "补贴",
    "租金",
    "底薪",
    "提成",
    "保险",
    "员工",
    "公司",
    "住宿",
    "着装",
    "管理",
]


@dataclass
class RiskScoringState:
    previous_smoothed: float | None = None
    consecutive_risk: int = 0
    alert_latched: bool = False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_scoring_mode(value: Any) -> str:
    mode = str(value or DEFAULT_SCORING_MODE).strip().lower()
    if mode not in {"baseline", "gated_v1"}:
        return DEFAULT_SCORING_MODE if DEFAULT_SCORING_MODE in {"baseline", "gated_v1"} else "gated_v1"
    return mode


def risk_level(score: float) -> str:
    if score >= 90:
        return "Critical"
    if score >= 70:
        return "High Risk"
    if score >= 50:
        return "Suspicious"
    return "Normal"


def final_label_from_score(score: float) -> str:
    return "Fraud Risk Detected" if score >= ALERT_THRESHOLD else "No High Risk Detected"


def has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def calibrated_text_score(raw_score: float, text: str) -> tuple[float, Dict[str, Any]]:
    text = text or ""
    has_action = has_any(text, FRAUD_ACTION_TERMS)
    has_urgency = has_any(text, URGENCY_TERMS)
    has_context = has_any(text, FRAUD_CONTEXT_TERMS)
    has_business = has_any(text, BUSINESS_FINANCE_TERMS)
    effective_len = len(text.strip())

    score = raw_score
    reason = "raw"
    if has_action:
        score = max(raw_score, 78.0 if has_urgency else 72.0)
        reason = "fraud_action"
    elif has_context and raw_score >= 85:
        score = max(raw_score, 76.0 if has_urgency else 70.0)
        reason = "fraud_context"
    elif effective_len < 12 and raw_score >= 70:
        score = min(raw_score, 50.0)
        reason = "short_text_cap"
    elif has_business and raw_score >= 70:
        score = min(raw_score, 62.0)
        reason = "business_context_cap"
    elif raw_score >= 90:
        score = min(raw_score, 68.0)
        reason = "unsupported_peak_cap"

    return round(score, 2), {
        "has_fraud_action": has_action,
        "has_urgency": has_urgency,
        "has_fraud_context": has_context,
        "has_business_context": has_business,
        "text_calibration_reason": reason,
        "original_text_score": round(raw_score, 2),
    }


def is_voice_strong(voice_score: float, case_type: str) -> bool:
    return case_type in {"synthetic_voice", "mixed_risk"} and voice_score >= 80


def score_window(
    raw_text_score: Any,
    voice_score: Any,
    text: str,
    state: RiskScoringState,
    scoring_mode: Any = None,
    case_type: str = "",
    text_weight: float = TEXT_WEIGHT,
    voice_weight: float = VOICE_WEIGHT,
    smoothing_previous_weight: float = SMOOTHING_PREVIOUS_WEIGHT,
) -> Dict[str, Any]:
    """Score one dynamic window and update state in-place."""
    mode = normalize_scoring_mode(scoring_mode)
    raw_text = round(safe_float(raw_text_score), 2)
    voice = safe_float(voice_score)
    raw_fused = round((text_weight * raw_text) + (voice_weight * voice), 2)
    current_weight = 1.0 - smoothing_previous_weight

    if mode == "baseline":
        smoothed = raw_fused if state.previous_smoothed is None else round(
            smoothing_previous_weight * state.previous_smoothed + current_weight * raw_fused,
            2,
        )
        state.previous_smoothed = smoothed
        state.consecutive_risk = state.consecutive_risk + 1 if raw_fused >= ALERT_THRESHOLD else 0
        state.alert_latched = state.alert_latched or smoothed >= ALERT_THRESHOLD
        return {
            "raw_text_score": raw_text,
            "text_score": raw_text,
            "voice_score": voice,
            "raw_fused_score": raw_fused,
            "fused_score": raw_fused,
            "smoothed_score": smoothed,
            "risk_level": risk_level(smoothed),
            "scoring_mode": "baseline",
            "consecutive_risk_windows": state.consecutive_risk,
            "alert_latched": state.alert_latched,
        }

    text_score, evidence = calibrated_text_score(raw_text, text)
    fused = round((text_weight * text_score) + (voice_weight * voice), 2)
    strong_voice = is_voice_strong(voice, case_type)
    if evidence["has_fraud_action"]:
        fused = max(fused, 74.0)
    elif evidence["has_fraud_context"] and (evidence["has_urgency"] or voice >= 60):
        fused = max(fused, 71.0)
    if strong_voice:
        fused = max(fused, 75.0)

    smoothed = fused if state.previous_smoothed is None else round(
        smoothing_previous_weight * state.previous_smoothed + current_weight * fused,
        2,
    )
    if strong_voice:
        smoothed = max(smoothed, 72.0)

    high_evidence = bool(evidence["has_fraud_action"]) or bool(evidence["has_fraud_context"]) or strong_voice
    state.consecutive_risk = state.consecutive_risk + 1 if fused >= ALERT_THRESHOLD else 0
    if high_evidence and smoothed >= ALERT_THRESHOLD:
        state.alert_latched = True
    if not high_evidence and state.consecutive_risk < 2 and smoothed >= ALERT_THRESHOLD:
        smoothed = 69.0
    elif state.alert_latched:
        smoothed = max(smoothed, 72.0)

    state.previous_smoothed = smoothed
    return {
        **evidence,
        "raw_text_score": raw_text,
        "text_score": text_score,
        "voice_score": voice,
        "raw_fused_score": raw_fused,
        "fused_score": fused,
        "smoothed_score": smoothed,
        "risk_level": risk_level(smoothed),
        "scoring_mode": "gated_v1",
        "consecutive_risk_windows": state.consecutive_risk,
        "voice_strong_evidence": strong_voice,
        "alert_latched": state.alert_latched,
    }
