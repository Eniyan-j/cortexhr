from datetime import date,datetime
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field 

# Attedance request 
class AttendanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attendance_feature_id: str
    employee_id: str
    month: date
    job_level: int
    scheduled_days: int
    present_days: int
    wfh_days: int
    half_days_unpaid: int
    paid_leave_days: int
    unpaid_absence_days: int
    longest_unpaid_streak_days: int
    late_arrivals: int
    avg_hours_worked: float
    overtime_hours: float
    attendance_rate: float

#Payroll request
class PayrollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payroll_item_id: str
    employee_id: str
    run_id: str
    month: date

    monthly_fixed_gross: float = Field(ge=0)
    payable_days: float = Field(ge=0)
    gross_pay: float = Field(ge=0)
    total_deductions: float = Field(ge=0)
    net_pay: float

    matching_payroll_record: bool = False
    duplicate_partner_id: str | None = None

#Common API response
class AnomalyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_id: str
    source_dataset: Literal[
        "attendance_monthly_features","payroll_monthly_features"]
    record_id: str
    employee_id: str
    period: date
    anomaly_score: float = Field(ge=0, le=1)
    flagged: Literal["Y", "N"]
    reason_code: str
    reason_text: str
    money_at_risk: float | None = Field(default=None, ge=0)
    model_version: str
    scored_at: datetime

    
# BATCH REQUEST SCHEMAS
class AttendanceBatchRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    records: list[AttendanceRequest] = Field(
        min_length=1,
        max_length=100
    )
class PayrollBatchRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    records: list[PayrollRequest] = Field(

        min_length=1,
        max_length=100
    )
# CHATBOT REQUEST SCHEMA
class ChatRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=2000
    )
# CHATBOT RESPONSE SCHEMA
class ChatResponse(BaseModel):

    model_config = ConfigDict(extra="forbid")

    intent: str
    reply: str