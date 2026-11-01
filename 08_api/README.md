# CortexHR Anomaly Detection API

## 1. Overview

FastAPI service for attendance and payroll anomaly detection.

## 2. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| POST | /v1/anomaly/attendance | Attendance anomaly detection |
| POST | /v1/anomaly/payroll | Payroll anomaly detection |

## 3. Model Artifacts

The service loads these artifacts once at startup:

- models/attendance_detector.joblib
- models/payroll_detector.joblib
- models/reason_codes.joblib

## 4. Install Dependencies

From the project root:

    pip install -r requirements.txt

## 5. Run the API

    cd /workspace/08_api
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

## 6. Test Health Endpoint

    curl -i http://127.0.0.1:8000/health

Verified: HTTP 200 OK.

## 7. Test Attendance Endpoint

    curl -i -X POST http://127.0.0.1:8000/v1/anomaly/attendance \
    -H "Content-Type: application/json" \
    -d '{"attendance_feature_id":"AF-00001","employee_id":"EMP-0002","month":"2025-04-01","job_level":4,"scheduled_days":20,"present_days":20,"wfh_days":18,"half_days_unpaid":0,"paid_leave_days":0,"unpaid_absence_days":0,"longest_unpaid_streak_days":0,"late_arrivals":1,"avg_hours_worked":8.5,"overtime_hours":0,"attendance_rate":1.0}'

Verified: HTTP 200 OK.

## 8. Test Payroll Endpoint

    curl -i -X POST http://127.0.0.1:8000/v1/anomaly/payroll \
    -H "Content-Type: application/json" \
    -d '{"payroll_item_id":"PI-TEST-001","employee_id":"EMP-TEST-001","run_id":"RUN-TEST-001","month":"2026-10-01","monthly_fixed_gross":50000,"payable_days":30,"gross_pay":50000,"total_deductions":5000,"net_pay":45000,"matching_payroll_record":false,"duplicate_partner_id":null}'

Verified: HTTP 200 OK.

## 9. Test Invalid Input

A payroll request missing net_pay was submitted.

Verified: HTTP 422 Unprocessable Entity.

## 10. Automated Tests

Run:

    python -m pytest /workspace/tests/test_api.py -v

Verified result:

    6 passed, 1 warning

Tests cover:
1. Health check.
2. Valid attendance request.
3. Invalid attendance request.
4. Valid payroll request.
5. Invalid payroll request.
6. Duplicate payroll detection.

## 11. Interactive API Documentation

While the service is running:

    http://127.0.0.1:8000/docs
