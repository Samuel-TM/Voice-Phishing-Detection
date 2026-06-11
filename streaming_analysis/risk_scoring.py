# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


TEXT_WEIGHT = 0.8
VOICE_WEIGHT = 0.2
SMOOTHING_PREVIOUS_WEIGHT = 0.65
ALERT_THRESHOLD = 70.0
SCORING_MODES = {"baseline", "gated_v1", "gated_v2", "gated_v3"}
DEFAULT_SCORING_MODE = os.environ.get("RISK_SCORING_MODE", "gated_v1").strip() or "gated_v1"
EVIDENCE_LOW_RISK = 0
EVIDENCE_SUSPICIOUS = 1
EVIDENCE_HIGH_RISK = 2
EVIDENCE_LEVEL_NAMES = {
    EVIDENCE_LOW_RISK: "low_risk",
    EVIDENCE_SUSPICIOUS: "suspicious",
    EVIDENCE_HIGH_RISK: "high_risk",
}

FRAUD_ACTION_TERMS = [
    "验证码",
    "验证马",
    "短信验证",
    "安全账户",
    "账户冻结",
    "冻结账户",
    "转账",
    "转入",
    "验证转账",
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
    "核验账号",
    "临时账户",
    "临时核验账户",
    "动态码",
    "共享验证码",
    "读给我",
    "告诉我验证码",
    "安全检测组件",
    "检测工具",
    "核验",
    "屏幕",
    "共享",
    "下载会议",
    "会议协助",
    "远程协助",
    "检测组件",
    "安全协助工具",
    "小额汇款",
    "注销额度",
    "账户确认",
    "到账通道",
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
    "安全风险",
    "异常标签",
    "异常记录",
    "支付账户",
    "退款通道",
    "系统验证",
    "风险拦截",
    "本人操作",
    "验证流程",
    "登录环境不安全",
    "检测组件",
    "安全协助工具",
    "风险记录",
    "征信查询",
    "贷款额度",
    "状态异常",
]

RISK_OBJECT_TERMS = [
    "账户",
    "账号",
    "资金",
    "身份",
    "实名",
    "银行",
    "银行卡",
    "验证码",
    "动态码",
    "交易",
    "登录",
    "支付",
    "退款",
    "冻结",
    "限制",
    "征信",
    "贷款",
    "核验",
    "验证",
    "风险",
    "安全",
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

HIGH_RISK_ACTION_TERMS = [
    "验证码",
    "验证马",
    "短信验证",
    "动态码",
    "共享验证码",
    "读给我",
    "告诉我验证码",
    "转账",
    "转入",
    "验证转账",
    "汇款",
    "小额汇款",
    "安全账户",
    "屏幕共享",
    "远程控制",
    "远程协助",
    "下载会议",
    "会议协助",
    "安全检测组件",
    "检测组件",
    "安全协助工具",
    "到账通道",
]

SUSPICIOUS_CONTEXT_TERMS = [
    *FRAUD_CONTEXT_TERMS,
    "异常账户",
    "账户异常",
    "冻结",
    "账户冻结",
    "冻结账户",
    "身份核验",
    "实名信息",
    "核验账户",
    "核验账号",
    "人工核验",
    "临时账户",
    "临时核验账户",
    "账户确认",
    "注销额度",
    "风险审核",
    "风险排查",
    "风控核查",
    "资金核验",
    "账户限制",
    "异常来源",
    "大额进出",
    "规定时间",
]


@dataclass
class RiskScoringState:
    previous_smoothed: float | None = None
    consecutive_risk: int = 0
    consecutive_suspicious_evidence: int = 0
    alert_latched: bool = False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_scoring_mode(value: Any) -> str:
    mode = str(value or DEFAULT_SCORING_MODE).strip().lower()
    if mode not in SCORING_MODES:
        return DEFAULT_SCORING_MODE if DEFAULT_SCORING_MODE in SCORING_MODES else "gated_v1"
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


def normalize_evidence_text(text: str) -> str:
    """Normalize ASR text before rule-based evidence matching."""
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff0-9])\s+(?=[\u4e00-\u9fff0-9])", "", text)


def has_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def matched_terms(text: str, terms: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(term for term in terms if term in text))


def _overlap_size(left: str, right: str) -> int:
    max_size = min(len(left), len(right))
    for size in range(max_size, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def deduplicate_recent_context(window_texts: Sequence[str]) -> str:
    """Deduplicate overlapping recent window texts while preserving order."""
    merged = ""
    for text in window_texts:
        cleaned = normalize_evidence_text(text)
        if not cleaned:
            continue
        if not merged:
            merged = cleaned
            continue
        if cleaned in merged:
            continue
        if merged in cleaned:
            merged = cleaned
            continue
        overlap = _overlap_size(merged, cleaned)
        if overlap:
            merged = f"{merged}{cleaned[overlap:]}"
        else:
            merged = f"{merged} {cleaned}"
    return merged.strip()


def combine_baseline_text_scores(raw_text_score: Any, context_text_score: Any) -> float:
    raw = safe_float(raw_text_score)
    context = safe_float(context_text_score)
    if raw >= 80.0:
        return round(max(raw, context), 2)
    return round(max(context, (0.7 * raw) + (0.3 * context)), 2)


def classify_text_evidence(text: str) -> Dict[str, Any]:
    evidence_text = normalize_evidence_text(text)
    high_matches = matched_terms(evidence_text, HIGH_RISK_ACTION_TERMS)
    suspicious_matches = matched_terms(evidence_text, SUSPICIOUS_CONTEXT_TERMS)
    business_matches = matched_terms(evidence_text, BUSINESS_FINANCE_TERMS)
    risk_object_matches = matched_terms(evidence_text, RISK_OBJECT_TERMS)
    urgency_matches = matched_terms(evidence_text, URGENCY_TERMS)

    if high_matches:
        level = EVIDENCE_HIGH_RISK
        reason = "high_risk_action"
    elif suspicious_matches:
        level = EVIDENCE_SUSPICIOUS
        reason = "suspicious_context"
    elif risk_object_matches and urgency_matches:
        level = EVIDENCE_SUSPICIOUS
        reason = "risk_object_with_urgency"
    elif business_matches:
        level = EVIDENCE_LOW_RISK
        reason = "business_context"
    else:
        level = EVIDENCE_LOW_RISK
        reason = "no_fraud_evidence"

    return {
        "evidence_text": evidence_text,
        "text_evidence_level": level,
        "text_evidence_level_name": EVIDENCE_LEVEL_NAMES[level],
        "text_evidence_reason": reason,
        "matched_high_risk_terms": high_matches,
        "matched_suspicious_terms": suspicious_matches,
        "matched_business_terms": business_matches,
        "matched_risk_object_terms": risk_object_matches,
        "matched_urgency_terms": urgency_matches,
        "has_fraud_action": bool(high_matches),
        "has_urgency": bool(urgency_matches),
        "has_fraud_context": bool(suspicious_matches),
        "has_risk_object": bool(risk_object_matches),
        "has_business_context": bool(business_matches),
    }


def calibrated_text_score(raw_score: float, text: str) -> tuple[float, Dict[str, Any]]:
    evidence_text = normalize_evidence_text(text)
    has_action = has_any(evidence_text, FRAUD_ACTION_TERMS)
    has_urgency = has_any(evidence_text, URGENCY_TERMS)
    has_context = has_any(evidence_text, FRAUD_CONTEXT_TERMS)
    has_business = has_any(evidence_text, BUSINESS_FINANCE_TERMS)
    effective_len = len(evidence_text.strip())

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
        "has_high_model_confidence": raw_score >= 95,
        "soft_text_evidence": False,
        "text_calibration_reason": reason,
        "original_text_score": round(raw_score, 2),
    }


def calibrated_text_score_v2(raw_score: float, text: str) -> tuple[float, Dict[str, Any]]:
    """Softer evidence gate: keep normal caps, but do not fully veto high-confidence fraud semantics."""
    evidence_text = normalize_evidence_text(text)
    has_action = has_any(evidence_text, FRAUD_ACTION_TERMS)
    has_urgency = has_any(evidence_text, URGENCY_TERMS)
    has_context = has_any(evidence_text, FRAUD_CONTEXT_TERMS)
    has_risk_object = has_any(evidence_text, RISK_OBJECT_TERMS)
    has_business = has_any(evidence_text, BUSINESS_FINANCE_TERMS)
    high_confidence = raw_score >= 95
    soft_evidence = high_confidence and has_context and (has_urgency or has_risk_object) and not has_business
    effective_len = len(evidence_text.strip())

    score = raw_score
    reason = "raw"
    if has_action:
        score = max(raw_score, 80.0 if has_urgency else 76.0)
        reason = "fraud_action"
    elif has_context and has_urgency:
        score = max(raw_score, 76.0)
        reason = "fraud_context_urgency"
    elif has_context and raw_score >= 85:
        score = max(raw_score, 72.0)
        reason = "fraud_context"
    elif soft_evidence:
        score = max(raw_score, 72.0)
        reason = "high_confidence_soft_evidence"
    elif effective_len < 12 and raw_score >= 70:
        score = min(raw_score, 50.0)
        reason = "short_text_cap"
    elif has_business and raw_score >= 70:
        score = min(raw_score, 62.0)
        reason = "business_context_cap"
    elif raw_score >= 95:
        score = min(raw_score, 70.0)
        reason = "unsupported_high_confidence_cap"
    elif raw_score >= 70 and not has_risk_object:
        score = min(raw_score, 64.0)
        reason = "unsupported_medium_confidence_cap"
    elif raw_score >= 90:
        score = min(raw_score, 68.0)
        reason = "unsupported_peak_cap"

    return round(score, 2), {
        "has_fraud_action": has_action,
        "has_urgency": has_urgency,
        "has_fraud_context": has_context,
        "has_risk_object": has_risk_object,
        "has_business_context": has_business,
        "has_high_model_confidence": high_confidence,
        "soft_text_evidence": soft_evidence,
        "text_calibration_reason": reason,
        "original_text_score": round(raw_score, 2),
    }


def calibrated_text_score_v3(raw_score: float, text: str) -> tuple[float, Dict[str, Any]]:
    """Evidence-layered gate: score strength is separated from evidence level."""
    evidence = classify_text_evidence(text)
    evidence_text = evidence["evidence_text"]
    evidence_level = evidence["text_evidence_level"]
    effective_len = len(evidence_text.strip())

    score = raw_score
    reason = "raw"
    if evidence_level == EVIDENCE_HIGH_RISK:
        score = max(raw_score, 82.0 if evidence["has_urgency"] else 78.0)
        reason = "high_risk_action"
    elif evidence_level == EVIDENCE_SUSPICIOUS:
        if raw_score >= 85:
            score = max(raw_score, 74.0)
            reason = "suspicious_context_high_confidence"
        else:
            score = max(raw_score, 66.0)
            reason = "suspicious_context"
    elif effective_len < 12 and raw_score >= 70:
        score = min(raw_score, 50.0)
        reason = "short_text_cap"
    elif evidence["has_business_context"] and raw_score >= 70:
        score = min(raw_score, 62.0)
        reason = "business_context_cap"
    elif raw_score >= 70:
        score = min(raw_score, 64.0)
        reason = "low_evidence_cap"

    return round(score, 2), {
        **{key: value for key, value in evidence.items() if key != "evidence_text"},
        "has_high_model_confidence": raw_score >= 95,
        "soft_text_evidence": False,
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
    context_text_score: Any = None,
    text_weight: float = TEXT_WEIGHT,
    voice_weight: float = VOICE_WEIGHT,
    smoothing_previous_weight: float = SMOOTHING_PREVIOUS_WEIGHT,
) -> Dict[str, Any]:
    """Score one dynamic window and update state in-place."""
    mode = normalize_scoring_mode(scoring_mode)
    raw_text = round(safe_float(raw_text_score), 2)
    context_text = round(safe_float(context_text_score, raw_text), 2)
    voice = safe_float(voice_score)
    raw_fused = round((text_weight * raw_text) + (voice_weight * voice), 2)
    current_weight = 1.0 - smoothing_previous_weight

    if mode == "baseline":
        text_score = combine_baseline_text_scores(raw_text, context_text)
        fused = round((text_weight * text_score) + (voice_weight * voice), 2)
        smoothed = fused if state.previous_smoothed is None else round(
            smoothing_previous_weight * state.previous_smoothed + current_weight * fused,
            2,
        )
        state.previous_smoothed = smoothed
        state.consecutive_risk = state.consecutive_risk + 1 if fused >= ALERT_THRESHOLD else 0
        state.alert_latched = state.alert_latched or smoothed >= ALERT_THRESHOLD
        return {
            "raw_text_score": raw_text,
            "raw_window_text_score": raw_text,
            "context_text_score": context_text,
            "text_score": text_score,
            "voice_score": voice,
            "raw_fused_score": raw_fused,
            "fused_score": fused,
            "smoothed_score": smoothed,
            "risk_level": risk_level(smoothed),
            "scoring_mode": "baseline",
            "consecutive_risk_windows": state.consecutive_risk,
            "alert_latched": state.alert_latched,
        }

    if mode == "gated_v3":
        text_score, evidence = calibrated_text_score_v3(raw_text, text)
    elif mode == "gated_v2":
        text_score, evidence = calibrated_text_score_v2(raw_text, text)
    else:
        text_score, evidence = calibrated_text_score(raw_text, text)
    fused = round((text_weight * text_score) + (voice_weight * voice), 2)
    strong_voice = is_voice_strong(voice, case_type)
    if mode == "gated_v3":
        text_evidence_level = safe_float(evidence.get("text_evidence_level"))
        if text_evidence_level >= EVIDENCE_HIGH_RISK:
            fused = max(fused, 76.0)
        elif text_evidence_level >= EVIDENCE_SUSPICIOUS:
            fused = max(fused, 70.0)
        if strong_voice:
            fused = max(fused, 75.0)
    elif mode == "gated_v2":
        if evidence["has_fraud_action"]:
            fused = max(fused, 76.0 if evidence["has_urgency"] else 73.0)
        elif evidence["has_fraud_context"] and evidence["has_urgency"]:
            fused = max(fused, 73.0)
        elif evidence["soft_text_evidence"]:
            fused = max(fused, 71.0)
        elif evidence["has_fraud_context"] and voice >= 60:
            fused = max(fused, 71.0)
        if strong_voice:
            fused = max(fused, 75.0)
    else:
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
    if mode in {"gated_v2", "gated_v3"} and evidence["has_fraud_action"] and fused >= 80:
        smoothed = max(smoothed, 72.0)
    if strong_voice:
        smoothed = max(smoothed, 72.0)

    if mode == "gated_v3":
        text_evidence_level = safe_float(evidence.get("text_evidence_level"))
        effective_evidence_level = max(text_evidence_level, EVIDENCE_HIGH_RISK if strong_voice else EVIDENCE_LOW_RISK)
        if effective_evidence_level >= EVIDENCE_SUSPICIOUS:
            state.consecutive_suspicious_evidence += 1
        else:
            state.consecutive_suspicious_evidence = 0
        state.consecutive_risk = state.consecutive_risk + 1 if fused >= ALERT_THRESHOLD else 0
        evidence_gate_passed = (
            effective_evidence_level >= EVIDENCE_HIGH_RISK
            or state.consecutive_suspicious_evidence >= 2
        )
        pre_gate_smoothed = smoothed
        if smoothed >= ALERT_THRESHOLD and evidence_gate_passed:
            state.alert_latched = True
        elif not state.alert_latched and smoothed >= ALERT_THRESHOLD:
            smoothed = 69.0
        elif state.alert_latched:
            smoothed = max(smoothed, 72.0)
        evidence.update({
            "effective_evidence_level": int(effective_evidence_level),
            "effective_evidence_level_name": EVIDENCE_LEVEL_NAMES[int(effective_evidence_level)],
            "evidence_gate_passed": evidence_gate_passed,
            "pre_gate_smoothed_score": round(pre_gate_smoothed, 2),
        })
    else:
        high_evidence = (
            bool(evidence["has_fraud_action"])
            or bool(evidence["has_fraud_context"])
            or bool(evidence.get("soft_text_evidence"))
            or strong_voice
        )
        state.consecutive_risk = state.consecutive_risk + 1 if fused >= ALERT_THRESHOLD else 0
        if high_evidence and smoothed >= ALERT_THRESHOLD:
            state.alert_latched = True
        if not high_evidence and not state.alert_latched and state.consecutive_risk < 2 and smoothed >= ALERT_THRESHOLD:
            smoothed = 69.0
        elif state.alert_latched:
            smoothed = max(smoothed, 72.0)

    state.previous_smoothed = smoothed
    return {
        **evidence,
        "raw_text_score": raw_text,
        "raw_window_text_score": raw_text,
        "context_text_score": context_text,
        "text_score": text_score,
        "voice_score": voice,
        "raw_fused_score": raw_fused,
        "fused_score": fused,
        "smoothed_score": smoothed,
        "risk_level": risk_level(smoothed),
        "scoring_mode": mode,
        "consecutive_risk_windows": state.consecutive_risk,
        "consecutive_suspicious_evidence_windows": state.consecutive_suspicious_evidence,
        "voice_strong_evidence": strong_voice,
        "alert_latched": state.alert_latched,
    }
