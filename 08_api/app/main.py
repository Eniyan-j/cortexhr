from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from uuid import uuid4

import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from .schemas import (
    AttendanceRequest,
    PayrollRequest,
    AnomalyResponse,
    PayrollBatchRequest,
    ChatRequest,
    ChatResponse,
    ResumeRankRequest,
    ResumeRankResponse
)
from .chatbot import CortexHRChatbot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
@asynccontextmanager
async def lifespan(app: FastAPI):

    attendance_path = MODELS_DIR / "attendance_detector.joblib"
    payroll_path = MODELS_DIR / "payroll_detector.joblib"
    reason_codes_path = MODELS_DIR / "reason_codes.joblib"
    chatbot_path = MODELS_DIR / "chatbot_bundle.joblib"
    resume_ranking_path = MODELS_DIR / "resume_ranking_bundle.joblib"
    
    required_files = [
        attendance_path,
        payroll_path,
        reason_codes_path,
        chatbot_path,
        resume_ranking_path
    ]
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {file}"
            )
    app.state.artifacts = {
        "attendance": joblib.load(attendance_path),
        "payroll": joblib.load(payroll_path),
        "reason_codes": joblib.load(reason_codes_path),
        "resume_ranking": joblib.load(resume_ranking_path)
    }
    app.state.chatbot = CortexHRChatbot(chatbot_path)
    print("Chatbot artifacts loaded successfully.")   
    print("All anomaly service artifacts loaded successfully.")
    yield
    #Application SHutdown 
    app.state.chatbot = None
    app.state.artifacts = {}
app = FastAPI(
    title="CortexHR Anomaly Service",
    description="Attendance and payroll anomaly detection API",
    version="1.0.0",
    lifespan=lifespan
)
@app.get("/health")
def health_check(request: Request):
    return {
        "status": "healthy",
        "loaded_artifacts": list(
            request.app.state.artifacts.keys()
        )
    }
@app.post(
    "/v1/anomaly/attendance",
    response_model=AnomalyResponse
)
def score_attendance(
    payload: AttendanceRequest,
    request: Request
):

    artifact = request.app.state.artifacts["attendance"]

    model = artifact["model"]

    features = pd.DataFrame(
        [
            {
                column: getattr(payload, column)
                for column in artifact["feature_columns"]
            }
        ],
        columns=artifact["feature_columns"]
    )

    medians = artifact["median_values"]

    features = features.fillna(medians)

    prediction = int(model.predict(features)[0])

    model_anomaly = prediction == -1

    raw_score = float(-model.decision_function(features)[0])

    score_min = artifact["score_min"]
    score_max = artifact["score_max"]

    if score_max == score_min:
        anomaly_score = 0.0
    else:
        anomaly_score = (
            (raw_score - score_min) / (score_max - score_min)
        )

    anomaly_score = max(0.0, min(1.0, anomaly_score))

    attendance_rate = payload.attendance_rate

    zero_attendance = attendance_rate == 0

    invalid_attendance = not (
        0 <= attendance_rate <= 1
    )

    rule_anomaly = zero_attendance or invalid_attendance

    flagged = model_anomaly or rule_anomaly
    
    reason_rules = (
        request.app.state.artifacts["reason_codes"]["attendance"]["rules"]
    )

    if not flagged:
        reason_code = "NO_ANOMALY"
        reason_text = "No attendance anomaly detected."

    elif (
        attendance_rate
        == reason_rules["NO_ATTENDANCE"]["value"]
    ):
        reason_code = "NO_ATTENDANCE"
        reason_text = "No presence recorded for the whole month"

    elif (
        payload.longest_unpaid_streak_days
        >= reason_rules["LONG_UNPAID_ABSENCE"]["value"]
    ):
        reason_code = "LONG_UNPAID_ABSENCE"
        reason_text = (
            f"{payload.longest_unpaid_streak_days} "
            "consecutive unpaid absence days"
        )

    elif (
        payload.late_arrivals
        >= reason_rules["FREQUENT_LATE"]["value"]
    ):
        reason_code = "FREQUENT_LATE"
        reason_text = (
            f"{payload.late_arrivals} late arrivals this month"
        )

    else:
        reason_code = "PATTERN_SHIFT"
        reason_text = (
            "Attendance pattern differs sharply from peers "
            "and own history"
        )

    return {
        "result_id": f"AN-{uuid4().hex[:8].upper()}",
        "source_dataset": "attendance_monthly_features",
        "record_id": payload.attendance_feature_id,
        "employee_id": payload.employee_id,
        "period": payload.month,
        "anomaly_score": anomaly_score,
        "flagged": "Y" if flagged else "N",
        "reason_code": reason_code,
        "reason_text": reason_text,
        "money_at_risk": None,
        "model_version": "iforest-attendance-0.1",
        "scored_at": datetime.now()
    }

@app.post(
    "/v1/anomaly/payroll",
    response_model=AnomalyResponse
)
def score_payroll(
    payload: PayrollRequest,
    request: Request
):

    artifact = request.app.state.artifacts["payroll"]

    model = artifact["model"]
    imputer = artifact["imputer"]
    feature_columns = artifact["feature_columns"]

    ratio_values = {
        "gross_fixed_ratio": (
            payload.gross_pay / payload.monthly_fixed_gross
            if payload.monthly_fixed_gross != 0
            else np.nan
        ),
        "deduction_ratio": (
            payload.total_deductions / payload.gross_pay
            if payload.gross_pay != 0
            else np.nan
        ),
        "net_ratio": (
            payload.net_pay / payload.gross_pay
            if payload.gross_pay != 0
            else np.nan
        )
    }

    X = pd.DataFrame(
        [[ratio_values[column] for column in feature_columns]],
        columns=feature_columns
    )
    X_imputed = imputer.transform(X)

    prediction = int(model.predict(X_imputed)[0])
    model_anomaly = prediction == -1

    raw_score = float(model.decision_function(X_imputed)[0])

    score_min = artifact["score_min"]
    score_max = artifact["score_max"]

    if score_max == score_min:
        anomaly_score = 0.0
    else:
        anomaly_score = (
            (score_max - raw_score) / (score_max - score_min)
        )

    anomaly_score = max(0.0, min(1.0, anomaly_score))

    negative_net_pay = payload.net_pay < 0

    zero_net_pay = (
        payload.net_pay == 0
        and payload.payable_days > 0
    )

    duplicate_payment = payload.matching_payroll_record

    flagged = (
        model_anomaly
        or negative_net_pay
        or zero_net_pay
        or duplicate_payment
    )

    calendar_days = pd.Timestamp(payload.month).days_in_month

    expected_fixed_pay = (
        payload.monthly_fixed_gross
        * payload.payable_days
        / calendar_days
    )

    gross = payload.gross_pay
    net = payload.net_pay

    money_at_risk = 0.0

    if not flagged:

        reason_code = "NO_ANOMALY"
        reason_text = "No payroll anomaly detected."

    elif negative_net_pay:

        reason_code = "NEGATIVE_NET_PAY"
        money_at_risk = abs(net)

        reason_text = (
            f"Deductions exceed gross pay by ₹{money_at_risk:,.2f}, "
            "resulting in negative net pay."
        )

    elif zero_net_pay:

        reason_code = "ZERO_PAY"
        money_at_risk = max(expected_fixed_pay - gross, 0)

        reason_text = (
            f"No net pay was recorded despite expected "
            f"prorated fixed pay of ₹{money_at_risk:,.2f}."
        )

    elif duplicate_payment:

        reason_code = "DOUBLE_PAYMENT"

        partner = payload.duplicate_partner_id or "matching record"

        is_additional = payload.payroll_item_id.startswith("PI-D")

        if is_additional:
            money_at_risk = max(net, 0)

            reason_text = (
                f"Additional record matches original payment {partner}; "
                f"verify whether ₹{money_at_risk:,.2f} was paid twice."
            )
        else:
            money_at_risk = 0.0

            reason_text = (
                f"This payment matches additional record {partner}; "
                "financial exposure is assigned to that record "
                "to avoid double-counting."
            )

    else:

        reason_code = "STATISTICAL_OUTLIER"

        money_at_risk = abs(gross - expected_fixed_pay)

        reason_text = (
            f"Gross pay differs from prorated fixed pay by "
            f"₹{money_at_risk:,.2f}; review whether overtime, variable "
            "pay or a salary change explains the difference."
        )

    return {
        "result_id": f"AN-{uuid4().hex[:8].upper()}",
        "source_dataset": "payroll_monthly_features",
        "record_id": payload.payroll_item_id,
        "employee_id": payload.employee_id,
        "period": payload.month,
        "anomaly_score": anomaly_score,
        "flagged": "Y" if flagged else "N",
        "reason_code": reason_code,
        "reason_text": reason_text,
        "money_at_risk": float(money_at_risk),
        "model_version": "iforest-payroll-0.1+rules",
        "scored_at": datetime.now()
    }

@app.post(
    "/v1/anomaly/payroll/batch",
    response_model=list[AnomalyResponse]
)
def score_payroll_batch(
    payload: PayrollBatchRequest,
    request: Request
):
    records = payload.records
    duplicate_groups = {}

    for index, record in enumerate(records):

        key = (
            record.employee_id,
            record.month,
            record.run_id,
            record.gross_pay,
            record.total_deductions,
            record.net_pay
        )

        duplicate_groups.setdefault(key, []).append(index)

    duplicate_updates = {}

    for indexes in duplicate_groups.values():

        if len(indexes) > 2:
            raise HTTPException(
                status_code=422,
                detail="Ambiguous duplicate group: more than two matching records."
            )

        if len(indexes) == 2:

            first, second = indexes

            first_is_additional = (
                records[first].payroll_item_id.startswith("PI-D")
            )

            second_is_additional = (
                records[second].payroll_item_id.startswith("PI-D")
            )

            if first_is_additional == second_is_additional:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot distinguish original and additional payment."
                )

            duplicate_updates[first] = {
                "matching_payroll_record": True,
                "duplicate_partner_id": (
                    records[second].payroll_item_id
                )
            }

            duplicate_updates[second] = {
                "matching_payroll_record": True,
                "duplicate_partner_id": (
                    records[first].payroll_item_id
                )
            }

    results = []

    for index, record in enumerate(records):

        if index in duplicate_updates:
            record = record.model_copy(
                update=duplicate_updates[index]
            )

        result = score_payroll(record, request)
        results.append(result)

    return results


# task 9: hr chatbot endpoint
@app.post(
    "/v1/chat",
    response_model=ChatResponse
)
def chat_endpoint(
    payload: ChatRequest,
    request: Request
):
    chatbot = request.app.state.chatbot
    result = chatbot.reply(payload.message)
    return {
        "intent": result.get("intent") or "unknown",
        "reply": result["reply"]
    }



# ============================================================
# TASK 11: RESUME RANKING ENDPOINT
# ============================================================

@app.post(
    "/v1/resumes/rank",
    response_model=ResumeRankResponse
)
def rank_resumes(
    payload: ResumeRankRequest,
    request: Request
):

    bundle = request.app.state.artifacts["resume_ranking"]

    job_id = payload.job_id

    # Validate job ID
    if job_id not in bundle["jobs"]:
        raise HTTPException(
            status_code=404,
            detail=f"Job ID not found: {job_id}"
        )

    job = bundle["jobs"][job_id]

    # Retrieve ranking records for the requested job
    job_rankings = bundle["rankings"][
        bundle["rankings"]["job_id"] == job_id
    ].copy()

    if job_rankings.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No candidate rankings available for {job_id}"
        )

    # Sort candidates by ranking score
    job_rankings = job_rankings.sort_values(
        "ranking_score",
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    # Prepare candidate profile lookup
    candidate_profiles = bundle["candidates"].set_index(
        "resume_id"
    )

    required_skills = set(job["required_skills"])
    preferred_skills = set(job["preferred_skills"])

    candidate_responses = []

    # Return the requested number of candidates
    for rank, (_, row) in enumerate(
        job_rankings.head(payload.top_k).iterrows(),
        start=1
    ):

        resume_id = row["resume_id"]

        if resume_id not in candidate_profiles.index:
            continue

        profile = candidate_profiles.loc[resume_id]

        candidate_skills = set(
            profile["extracted_skills"]
        )

        matched_required = sorted(
            candidate_skills.intersection(required_skills)
        )

        missing_required = sorted(
            required_skills.difference(candidate_skills)
        )

        matched_preferred = sorted(
            candidate_skills.intersection(preferred_skills)
        )

        # Explain the scoring components
        explanations = [
            (
                f"Required skills matched: "
                f"{len(matched_required)} of {len(required_skills)}."
            ),
            (
                f"Preferred skills matched: "
                f"{len(matched_preferred)} of {len(preferred_skills)}."
            ),
            (
                f"Candidate experience: "
                f"{float(profile['experience_years']):.1f} years; "
                f"job minimum: {job['minimum_experience_years']:.1f} years."
            ),
            (
                f"Text similarity contributed "
                f"{float(row['tfidf_similarity']) * 100:.2f}% similarity."
            )
        ]

        candidate_responses.append({

            "rank": rank,

            "resume_id": str(resume_id),

            "candidate_title": str(
                profile["candidate_title"]
            ),

            "experience_years": float(
                profile["experience_years"]
            ),

            "extracted_skills": sorted(candidate_skills),

            "ranking_score": float(
                row["ranking_score"]
            ),

            "tfidf_similarity": float(
                row["tfidf_similarity"]
            ),

            "required_skill_coverage": float(
                row["required_skill_coverage"]
            ),

            "preferred_skill_coverage": float(
                row["preferred_skill_coverage"]
            ),

            "experience_relevance": float(
                row["experience_relevance"]
            ),

            "matched_required_skills": matched_required,

            "missing_required_skills": missing_required,

            "matched_preferred_skills": matched_preferred,

            "explanation": explanations
        })


    return {

        "job_id": job_id,

        "job_title": job["job_title"],

        "model_version": bundle["model_version"],

        "total_candidates": len(job_rankings),

        "returned_candidates": len(candidate_responses),

        "scoring_weights": bundle["scoring_weights"],

        "candidates": candidate_responses,

        "score_interpretation": (
            "Ranking score is a weighted matching score out of 100, "
            "not a probability of hiring. It supports recruiter review "
            "and must not replace human judgement."
        )
    }

