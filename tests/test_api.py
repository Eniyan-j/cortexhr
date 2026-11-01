from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Make the API application importable
API_DIR = Path(__file__).resolve().parents[1] / "08_api"
sys.path.insert(0, str(API_DIR))

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def attendance_payload():
    return {
        "attendance_feature_id": "AF-TEST-001",
        "employee_id": "EMP-TEST-001",
        "month": "2026-10-01",
        "job_level": 4,
        "scheduled_days": 20,
        "present_days": 18,
        "wfh_days": 0,
        "half_days_unpaid": 0,
        "paid_leave_days": 0,
        "unpaid_absence_days": 2,
        "longest_unpaid_streak_days": 2,
        "late_arrivals": 1,
        "avg_hours_worked": 8.0,
        "overtime_hours": 0.0,
        "attendance_rate": 0.9
    }


def payroll_payload():
    return {
        "payroll_item_id": "PI-TEST-001",
        "employee_id": "EMP-TEST-001",
        "run_id": "RUN-TEST-001",
        "month": "2026-10-01",
        "monthly_fixed_gross": 50000.0,
        "payable_days": 30.0,
        "gross_pay": 50000.0,
        "total_deductions": 5000.0,
        "net_pay": 45000.0,
        "matching_payroll_record": False,
        "duplicate_partner_id": None
    }


OFFICIAL_FIELDS = {
    "result_id",
    "source_dataset",
    "record_id",
    "employee_id",
    "period",
    "anomaly_score",
    "flagged",
    "reason_code",
    "reason_text",
    "money_at_risk",
    "model_version",
    "scored_at"
}


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200

    assert set(response.json()["loaded_artifacts"]) == {
        "attendance",
        "payroll",
        "reason_codes"
    }


def test_attendance_valid_request(client):
    response = client.post(
        "/v1/anomaly/attendance",
        json=attendance_payload()
    )

    assert response.status_code == 200
    assert set(response.json()) == OFFICIAL_FIELDS
    assert response.json()["source_dataset"] == (
        "attendance_monthly_features"
    )


def test_attendance_missing_field_returns_422(client):
    payload = attendance_payload()
    payload.pop("attendance_rate")

    response = client.post(
        "/v1/anomaly/attendance",
        json=payload
    )

    assert response.status_code == 422


def test_payroll_valid_request(client):
    response = client.post(
        "/v1/anomaly/payroll",
        json=payroll_payload()
    )

    assert response.status_code == 200
    assert set(response.json()) == OFFICIAL_FIELDS
    assert response.json()["source_dataset"] == (
        "payroll_monthly_features"
    )


def test_payroll_missing_field_returns_422(client):
    payload = payroll_payload()
    payload.pop("net_pay")

    response = client.post(
        "/v1/anomaly/payroll",
        json=payload
    )

    assert response.status_code == 422


def test_payroll_duplicate_detection(client):
    payload = payroll_payload()

    payload["payroll_item_id"] = "PI-D-TEST-001"
    payload["matching_payroll_record"] = True
    payload["duplicate_partner_id"] = "PI-ORIGINAL-001"

    response = client.post(
        "/v1/anomaly/payroll",
        json=payload
    )

    assert response.status_code == 200
    assert response.json()["reason_code"] == "DOUBLE_PAYMENT"
    assert response.json()["money_at_risk"] == 45000.0
