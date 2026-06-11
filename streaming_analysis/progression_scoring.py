# -*- coding: utf-8 -*-
"""Progression-aware early-warning scoring for simulated streaming analysis.

The module keeps raw model scores intact and separates them from the alert
decision. Keyword matches are only candidates; High Risk requires confirmed
evidence events and stage progression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Sequence


ALERT_THRESHOLD = 70.0
DEFAULT_PROGRESSION_MODE = os.environ.get("PROGRESSION_SCORING_MODE", "progression_v1").strip() or "progression_v1"
PROGRESSION_MODES = {"progression_v1"}

SCAM_STAGES = {
    "benign_or_unknown",
    "risk_context_setup",
    "pressure_or_isolation",
    "harmful_action_request",
}
VOICE_STAGES = {"normal_voice", "synthetic_voice_suspected"}
ALERT_DECISIONS = {"Observe", "Suspicious", "High Risk", "Critical"}

RISK_CONTEXT_TERMS = [
    "异常交易",
    "异常登录",
    "异常资金",
    "异常记录",
    "风险记录",
    "风险交易",
    "风险处理",
    "风险拦截",
    "账户异常",
    "账号异常",
    "登录异常",
    "支付异常",
    "状态异常",
    "退款异常",
    "退款通道",
    "到账通道",
    "身份核验",
    "身份核查",
    "人工核验",
    "本人操作",
    "安全验证",
    "系统验证",
    "资金来源",
    "临时限制",
    "临时冻结",
    "账户冻结",
    "账户安全",
    "安全风险",
    "登录环境不安全",
    "涉案",
    "洗钱",
]

RISK_PROCESS_TERMS = [
    "核验",
    "核查",
    "验证",
    "确认",
    "处理",
    "解除",
    "恢复",
    "拦截",
    "冻结",
    "限制",
    "审核",
    "人工",
    "流程",
]

PRESSURE_PHRASES = [
    "不要挂断",
    "不要挂",
    "不要告诉",
    "不要向其他人透露",
    "不要透露",
    "不要关闭",
    "不要退出",
    "保持通话",
    "保密",
    "不能中断",
    "流程不能中断",
]

URGENCY_TERMS = [
    "马上",
    "立即",
    "现在",
    "尽快",
    "否则",
    "逾期",
    "十分钟",
    "一分钟",
    "今天",
    "规定时间",
]

HARMFUL_OBJECT_TERMS = [
    "验证码",
    "验证马",
    "动态码",
    "短信验证",
    "登录确认码",
    "银行卡",
    "银行卡信息",
    "卡号",
    "密码",
    "安全账户",
    "核验账户",
    "核验账号",
    "临时账户",
    "临时核验账户",
    "指定金额",
    "确认金额",
    "小额汇款",
    "转账",
    "转入",
    "汇款",
    "屏幕共享",
    "屏幕共亨",
    "共享屏幕",
    "远程控制",
    "远程协助",
    "安全协助工具",
    "检测组件",
    "检测工具",
]

HARMFUL_ACTION_TERMS = [
    "告诉我",
    "发给我",
    "读给我",
    "念给我",
    "共享",
    "提供",
    "提交",
    "输入",
    "打开",
    "下载",
    "安装",
    "开启",
    "转入",
    "转账",
    "汇款",
    "完成",
    "按照我说",
    "按我说",
    "按提示",
    "操作",
]

REQUEST_TERMS = [
    "请您",
    "需要您",
    "麻烦您",
    "请",
    "按照",
    "配合",
    "帮您",
    "我会",
]

HIGH_DANGER_TERMS = [
    "验证码",
    "验证马",
    "动态码",
    "安全账户",
    "临时账户",
    "核验账户",
    "核验账号",
    "转账",
    "转入",
    "汇款",
    "屏幕共享",
    "远程控制",
    "银行卡信息",
]

BENIGN_FINANCE_TERMS = [
    "工资",
    "薪资",
    "底薪",
    "提成",
    "社保",
    "五险一金",
    "保险",
    "医疗保险",
    "合同",
    "维修合同",
    "支付宝",
    "微信支付",
    "花呗",
    "借呗",
    "贷款",
    "还款",
    "付款",
    "生活",
    "学生",
    "班费",
    "奖学金",
    "助学金",
    "彩礼",
    "新闻",
    "工资单",
]

BENIGN_DISCUSSION_TERMS = [
    "我觉得",
    "你觉得",
    "据我了解",
    "讨论",
    "方案",
    "建议",
    "新闻",
    "听说",
    "故事",
    "公司",
    "学生",
    "学校",
]


@dataclass
class TextSegment:
    text: str
    start_sec: float
    end_sec: float


@dataclass
class ProgressionScoringState:
    recent_text_segments: List[TextSegment] = field(default_factory=list)
    evidence_memory: List[Dict[str, Any]] = field(default_factory=list)
    scam_stage: str = "benign_or_unknown"
    alert_decision: str = "Observe"
    alert_latched: bool = False
    high_risk_since_sec: float | None = None
    previous_display_score: float | None = None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_progression_mode(value: Any) -> str:
    mode = str(value or DEFAULT_PROGRESSION_MODE).strip()
    return mode if mode in PROGRESSION_MODES else "progression_v1"


def final_label_from_decision(decision: str) -> str:
    if decision in {"High Risk", "Critical"}:
        return "Fraud Risk Detected"
    return "No High Risk Detected"


def risk_level_from_decision(decision: str) -> str:
    if decision == "Critical":
        return "Critical"
    if decision == "High Risk":
        return "High Risk"
    if decision == "Suspicious":
        return "Suspicious"
    return "Normal"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _matched_terms(text: str, terms: Sequence[str]) -> List[str]:
    normalized = _normalize_text(text)
    return [term for term in terms if term and term in normalized]


def _has_any(text: str, terms: Sequence[str]) -> bool:
    return bool(_matched_terms(text, terms))


def _is_near_duplicate(new_text: str, existing_text: str) -> bool:
    new_norm = _normalize_text(new_text)
    old_norm = _normalize_text(existing_text)
    if not new_norm or not old_norm:
        return False
    if new_norm in old_norm or old_norm in new_norm:
        shorter = min(len(new_norm), len(old_norm))
        longer = max(len(new_norm), len(old_norm))
        return shorter / max(longer, 1) >= 0.55
    overlap = len(set(new_norm) & set(old_norm))
    return overlap / max(min(len(set(new_norm)), len(set(old_norm))), 1) >= 0.86


def update_short_context(
    state: ProgressionScoringState,
    text: str,
    start_sec: float,
    end_sec: float,
    max_seconds: float = 30.0,
    max_segments: int = 3,
) -> str:
    clean_text = str(text or "").strip()
    if clean_text:
        is_duplicate = any(_is_near_duplicate(clean_text, segment.text) for segment in state.recent_text_segments)
        if not is_duplicate:
            state.recent_text_segments.append(TextSegment(clean_text, start_sec, end_sec))

    cutoff = end_sec - max_seconds
    state.recent_text_segments = [
        segment for segment in state.recent_text_segments
        if segment.end_sec >= cutoff and segment.text.strip()
    ][-max_segments:]
    return " ".join(segment.text for segment in state.recent_text_segments).strip()


def _event_payload(
    event_type: str,
    status: str,
    matched_terms: Sequence[str],
    reason: str,
    timestamp_sec: float,
) -> Dict[str, Any]:
    return {
        "event_type": event_type,
        "status": status,
        "matched_terms": list(dict.fromkeys(matched_terms)),
        "reason": reason,
        "timestamp_sec": round(timestamp_sec, 2),
    }


def extract_evidence_events(text: str, timestamp_sec: float) -> Dict[str, Any]:
    """Extract candidate and confirmed evidence events from short context."""
    normalized_text = _normalize_text(text)
    context_terms = _matched_terms(normalized_text, RISK_CONTEXT_TERMS)
    process_terms = _matched_terms(normalized_text, RISK_PROCESS_TERMS)
    pressure_phrases = _matched_terms(normalized_text, PRESSURE_PHRASES)
    urgency_terms = _matched_terms(normalized_text, URGENCY_TERMS)
    harmful_objects = _matched_terms(normalized_text, HARMFUL_OBJECT_TERMS)
    harmful_actions = _matched_terms(normalized_text, HARMFUL_ACTION_TERMS)
    request_terms = _matched_terms(normalized_text, REQUEST_TERMS)
    benign_finance_terms = _matched_terms(normalized_text, BENIGN_FINANCE_TERMS)
    benign_discussion_terms = _matched_terms(normalized_text, BENIGN_DISCUSSION_TERMS)

    events: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    benign_finance_context = bool(benign_finance_terms and not pressure_phrases and not harmful_actions)
    if benign_finance_terms and benign_discussion_terms and not pressure_phrases:
        benign_finance_context = True

    if context_terms:
        if process_terms and not benign_finance_context:
            events.append(_event_payload(
                "risk_context_setup",
                "confirmed",
                context_terms + process_terms,
                "risk context with verification/process semantics",
                timestamp_sec,
            ))
        else:
            candidates.append(_event_payload(
                "risk_context_setup",
                "candidate",
                context_terms,
                "risk context term without enough process semantics",
                timestamp_sec,
            ))

    if pressure_phrases:
        events.append(_event_payload(
            "pressure_or_isolation",
            "confirmed",
            pressure_phrases,
            "explicit isolation or call-control pressure",
            timestamp_sec,
        ))
    elif urgency_terms:
        if request_terms and not benign_finance_context:
            events.append(_event_payload(
                "pressure_or_isolation",
                "confirmed",
                urgency_terms + request_terms,
                "urgency appears in a request/instruction context",
                timestamp_sec,
            ))
        else:
            candidates.append(_event_payload(
                "pressure_or_isolation",
                "candidate",
                urgency_terms,
                "urgency term without enough request context",
                timestamp_sec,
            ))

    if harmful_objects:
        has_user_instruction = bool(request_terms and harmful_actions)
        has_action_phrase = bool(harmful_actions and ("请" in normalized_text or "需要" in normalized_text or "按照" in normalized_text))
        if (has_user_instruction or has_action_phrase) and not (benign_finance_context and not pressure_phrases):
            events.append(_event_payload(
                "harmful_action_request",
                "confirmed",
                harmful_objects + harmful_actions + request_terms,
                "dangerous object appears with request/instruction intent",
                timestamp_sec,
            ))
        else:
            candidates.append(_event_payload(
                "harmful_action_request",
                "candidate",
                harmful_objects,
                "dangerous object term without confirmed user instruction",
                timestamp_sec,
            ))

    return {
        "confirmed_events": events,
        "candidate_events": candidates,
        "benign_finance_context": benign_finance_context,
        "matched_context_terms": context_terms,
        "matched_pressure_terms": pressure_phrases + urgency_terms,
        "matched_harmful_terms": harmful_objects + harmful_actions,
        "matched_benign_finance_terms": benign_finance_terms,
    }


def _recent_confirmed(memory: Sequence[Dict[str, Any]], event_type: str, current_sec: float, window_sec: float) -> bool:
    for event in reversed(memory):
        if event.get("event_type") != event_type or event.get("status") != "confirmed":
            continue
        if current_sec - safe_float(event.get("timestamp_sec")) <= window_sec:
            return True
    return False


def _append_confirmed_events(state: ProgressionScoringState, events: Sequence[Dict[str, Any]], current_sec: float) -> None:
    for event in events:
        signature = (
            event.get("event_type"),
            tuple(event.get("matched_terms", [])[:4]),
            round(safe_float(event.get("timestamp_sec")), 1),
        )
        exists = any(
            (
                item.get("event_type"),
                tuple(item.get("matched_terms", [])[:4]),
                round(safe_float(item.get("timestamp_sec")), 1),
            ) == signature
            for item in state.evidence_memory
        )
        if not exists:
            state.evidence_memory.append(dict(event))
    state.evidence_memory = [
        event for event in state.evidence_memory
        if current_sec - safe_float(event.get("timestamp_sec")) <= 90.0
    ]


def update_scam_stage(
    state: ProgressionScoringState,
    confirmed_events: Sequence[Dict[str, Any]],
    current_sec: float,
) -> str:
    event_types = {event.get("event_type") for event in confirmed_events}
    has_context = "risk_context_setup" in event_types or _recent_confirmed(state.evidence_memory, "risk_context_setup", current_sec, 45.0)
    has_pressure = "pressure_or_isolation" in event_types or _recent_confirmed(state.evidence_memory, "pressure_or_isolation", current_sec, 45.0)
    has_harmful = "harmful_action_request" in event_types

    if has_harmful:
        state.scam_stage = "harmful_action_request"
    elif has_context and has_pressure:
        state.scam_stage = "pressure_or_isolation"
    elif has_context:
        state.scam_stage = "risk_context_setup"
    elif state.scam_stage not in SCAM_STAGES:
        state.scam_stage = "benign_or_unknown"
    return state.scam_stage


def decide_alert(
    state: ProgressionScoringState,
    raw_window_text_score: float,
    short_context_text_score: float,
    confirmed_events: Sequence[Dict[str, Any]],
    candidate_events: Sequence[Dict[str, Any]],
    benign_finance_context: bool,
    current_sec: float,
) -> tuple[str, str]:
    event_types = {event.get("event_type") for event in confirmed_events}
    has_harmful_now = "harmful_action_request" in event_types
    has_context_recent = "risk_context_setup" in event_types or _recent_confirmed(
        state.evidence_memory, "risk_context_setup", current_sec, 45.0
    )
    has_pressure_recent = "pressure_or_isolation" in event_types or _recent_confirmed(
        state.evidence_memory, "pressure_or_isolation", current_sec, 45.0
    )
    has_high_danger = any(
        term in HIGH_DANGER_TERMS
        for event in confirmed_events
        for term in event.get("matched_terms", [])
    )
    repeated_harmful_after_alert = (
        state.alert_latched
        and has_harmful_now
        and state.high_risk_since_sec is not None
        and current_sec - state.high_risk_since_sec <= 30.0
    )

    if repeated_harmful_after_alert or (has_harmful_now and has_high_danger and state.alert_latched):
        return "Critical", "confirmed harmful action repeated after high-risk alert"

    if has_harmful_now and (has_context_recent or has_pressure_recent or short_context_text_score >= 70.0):
        return "High Risk", "confirmed harmful action with progression or high short-context score"

    if has_context_recent and has_pressure_recent and short_context_text_score >= 85.0:
        return "High Risk", "context and pressure progressed within 45 seconds with high short-context score"

    if confirmed_events:
        return "Suspicious", "confirmed evidence exists but high-risk conditions are not met"

    if candidate_events:
        return "Suspicious", "candidate evidence only; keyword candidates cannot trigger high risk"

    if raw_window_text_score >= 70.0 or short_context_text_score >= 70.0:
        if benign_finance_context:
            return "Suspicious", "high text score in benign finance context without confirmed evidence"
        return "Suspicious", "high text score without confirmed progression evidence"

    return "Observe", "no confirmed progression evidence"


def _display_score(decision: str, raw_score: float, short_score: float, previous: float | None) -> float:
    basis = max(raw_score, short_score)
    if decision == "Critical":
        score = max(90.0, min(100.0, basis))
    elif decision == "High Risk":
        score = max(72.0, min(89.0, basis))
    elif decision == "Suspicious":
        score = max(50.0, min(69.0, basis if basis > 0 else 55.0))
    else:
        score = min(49.0, basis)

    if previous is not None and decision in {"High Risk", "Critical"}:
        score = max(score, min(89.0 if decision == "High Risk" else 100.0, previous))
    return round(score, 2)


def _stage_confidence(stage: str, confirmed_events: Sequence[Dict[str, Any]], short_score: float) -> float:
    if stage == "harmful_action_request":
        base = 0.9
    elif stage == "pressure_or_isolation":
        base = 0.72
    elif stage == "risk_context_setup":
        base = 0.62
    else:
        base = 0.25
    base += min(len(confirmed_events), 3) * 0.04
    if short_score >= 85:
        base += 0.06
    elif short_score >= 70:
        base += 0.03
    return round(min(base, 0.99), 2)


def score_progression_window(
    raw_window_text_score: Any,
    short_context_text_score: Any,
    voice_score: Any,
    text: str,
    short_context_text: str,
    state: ProgressionScoringState,
    start_sec: float,
    end_sec: float,
    scoring_mode: Any = None,
) -> Dict[str, Any]:
    mode = normalize_progression_mode(scoring_mode)
    raw_score = round(safe_float(raw_window_text_score), 2)
    short_score = round(safe_float(short_context_text_score), 2)
    voice = round(safe_float(voice_score), 2)
    current_sec = safe_float(end_sec)
    voice_stage = "synthetic_voice_suspected" if voice >= 80.0 else "normal_voice"

    evidence = extract_evidence_events(short_context_text or text, current_sec)
    confirmed_events = evidence["confirmed_events"]
    candidate_events = evidence["candidate_events"]
    _append_confirmed_events(state, confirmed_events, current_sec)
    scam_stage = update_scam_stage(state, confirmed_events, current_sec)
    decision, reason = decide_alert(
        state=state,
        raw_window_text_score=raw_score,
        short_context_text_score=short_score,
        confirmed_events=confirmed_events,
        candidate_events=candidate_events,
        benign_finance_context=bool(evidence["benign_finance_context"]),
        current_sec=current_sec,
    )
    if decision in {"High Risk", "Critical"}:
        state.alert_latched = True
        if state.high_risk_since_sec is None:
            state.high_risk_since_sec = current_sec
    state.alert_decision = decision

    display_score = _display_score(decision, raw_score, short_score, state.previous_display_score)
    state.previous_display_score = display_score

    return {
        "raw_window_text_score": raw_score,
        "short_context_text_score": short_score,
        "text_score": raw_score,
        "voice_score": voice,
        "voice_stage": voice_stage,
        "voice_evidence": {
            "voice_stage": voice_stage,
            "voice_score": voice,
            "synthetic_voice_suspected": voice_stage == "synthetic_voice_suspected",
        },
        "scam_stage": scam_stage,
        "stage_confidence": _stage_confidence(scam_stage, confirmed_events, short_score),
        "evidence_events": confirmed_events,
        "candidate_evidence_events": candidate_events,
        "evidence_memory": list(state.evidence_memory),
        "benign_finance_context": bool(evidence["benign_finance_context"]),
        "alert_decision": decision,
        "alert_reason": reason,
        "fused_score": display_score,
        "smoothed_score": display_score,
        "display_score": display_score,
        "risk_level": risk_level_from_decision(decision),
        "final_label": final_label_from_decision(decision),
        "scoring_mode": mode,
        "alert_latched": state.alert_latched,
        "matched_context_terms": evidence["matched_context_terms"],
        "matched_pressure_terms": evidence["matched_pressure_terms"],
        "matched_harmful_terms": evidence["matched_harmful_terms"],
        "matched_benign_finance_terms": evidence["matched_benign_finance_terms"],
    }
