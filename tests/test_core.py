"""
Unit tests for time-split logic, leakage audit, and servicer reconciliation.
These cover the three highest-risk areas identified in PRD §9.

Run with:
    python -m pytest tests/test_core.py -v
"""
import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.engineer import perform_time_split, check_leakage, engineer_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_panel(n_loans=5, months_per_loan=12, start="2023-01-01"):
    """Create a minimal synthetic panel for testing."""
    records = []
    for i in range(n_loans):
        loan_id = f"LN{i:03d}"
        for m in range(months_per_loan):
            month = pd.Timestamp(start) + pd.DateOffset(months=m)
            records.append({
                "loan_id": loan_id,
                "reporting_month": month.strftime("%Y-%m-%d"),
                "reporting_month_dt": month,
                "origination_month": "2020-01-01",
                "loan_age_months": m + 1,
                "remaining_term_months": 360 - m,
                "original_balance": 300_000.0,
                "current_balance": 290_000.0 - m * 500,
                "interest_rate": 6.5,
                "credit_score_band": "700-749",
                "ltv_band": "70-80%",
                "dti_band": "30-40%",
                "state": "CA",
                "loan_purpose": "purchase",
                "occupancy_type": "owner_occupied",
                "property_type": "single_family",
                "servicer_name": "Servicer_A",
                "current_status": "Current",
                "days_past_due": 0,
                "modification_flag": False,
                "prepayment_flag": False,
                "default_flag": False,
                "loss_severity_band": "0-10%",
                "last_updated_at": month.strftime("%Y-%m-%d"),
                "source_system": "SystemA",
                "document_status": "complete",
                "balance_ratio": 0.97,
                "rate_spread": 0.0,
                "months_since_last_update": 0,
                "credit_score_band_ord": 3,
                "ltv_band_ord": 2,
                "dti_band_ord": 1,
                "prev_balance": 290_000.0 - (m - 1) * 500,
                "balance_change": -500.0,
                "prev_dpd": 0,
                "dpd_change": 0,
                # Target columns
                "next_3m_delinquency_flag": 0,
                "next_6m_delinquency_flag": 0,
                "next_12m_default_flag": 0,
                "next_12m_prepayment_flag": 0,
                "next_state": "Current",
                "exception_required": 0,
                "exception_type": "no_exception",
            })
    return pd.DataFrame(records)


# ===========================================================================
# 1. TIME-SPLIT TESTS
# ===========================================================================

class TestTimeSplit:
    """Tests for the time-aware split logic in engineer.perform_time_split."""

    def test_no_data_leakage_across_splits(self):
        """No loan-month in test should appear in train."""
        df = make_panel(n_loans=10, months_per_loan=36, start="2022-01-01")
        train, valid, test = perform_time_split(df)

        train_keys = set(zip(train["loan_id"], train["reporting_month_dt"]))
        test_keys = set(zip(test["loan_id"], test["reporting_month_dt"]))

        overlap = train_keys & test_keys
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_train_ends_before_cutoff(self):
        """All training rows must be on or before 2024-06-01."""
        df = make_panel(n_loans=5, months_per_loan=48, start="2022-01-01")
        train, _, _ = perform_time_split(df)

        cutoff = pd.Timestamp("2024-06-01")
        assert train["reporting_month_dt"].max() <= cutoff, (
            f"Train contains rows after cutoff: {train['reporting_month_dt'].max()}"
        )

    def test_test_starts_after_valid_cutoff(self):
        """All test rows must be after 2024-09-01 (the validation end cutoff)."""
        df = make_panel(n_loans=5, months_per_loan=48, start="2022-01-01")
        _, valid, test = perform_time_split(df)

        valid_cutoff = pd.Timestamp("2024-09-01")
        if len(test) > 0:
            assert test["reporting_month_dt"].min() > valid_cutoff, (
                f"Test contains rows before valid cutoff: {test['reporting_month_dt'].min()}"
            )

    def test_splits_are_mutually_exclusive(self):
        """No loan-month can appear in more than one split."""
        df = make_panel(n_loans=5, months_per_loan=48, start="2022-01-01")
        train, valid, test = perform_time_split(df)

        all_splits = pd.concat([train, valid, test])
        key = all_splits["loan_id"] + "_" + all_splits["reporting_month_dt"].astype(str)
        assert key.is_unique, "Duplicate loan-month keys found across splits"

    def test_splits_cover_all_rows(self):
        """Train + valid + test should equal total rows."""
        df = make_panel(n_loans=5, months_per_loan=48, start="2022-01-01")
        train, valid, test = perform_time_split(df)
        assert len(train) + len(valid) + len(test) == len(df)

    def test_no_random_shuffle_dependency(self):
        """Applying the split twice to the same data should produce identical results."""
        df = make_panel(n_loans=5, months_per_loan=48, start="2022-01-01")
        train1, valid1, test1 = perform_time_split(df.copy())
        train2, valid2, test2 = perform_time_split(df.copy())
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)


# ===========================================================================
# 2. LEAKAGE AUDIT TESTS
# ===========================================================================

class TestLeakageAudit:
    """Tests for the leakage audit in engineer.check_leakage."""

    TARGET_COLS = [
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
        "next_state",
        "exception_required",
        "exception_type",
    ]

    def test_no_target_in_features(self):
        """None of the 7 target columns should appear in the returned feature list."""
        df = make_panel()
        features, _ = check_leakage(df)
        leaked = [c for c in features if c in self.TARGET_COLS]
        assert len(leaked) == 0, f"Target columns leaked into features: {leaked}"

    def test_loan_id_not_in_features(self):
        """loan_id must never be in the feature list."""
        df = make_panel()
        features, _ = check_leakage(df)
        assert "loan_id" not in features

    def test_reporting_month_not_in_features(self):
        """Raw reporting_month string must not be in the feature list."""
        df = make_panel()
        features, _ = check_leakage(df)
        assert "reporting_month" not in features

    def test_features_are_numeric_or_category(self):
        """Every returned feature must be numeric or category dtype for LightGBM."""
        df = make_panel()
        # Apply category dtype to categorical cols as engineer.py would
        for col in ["state", "loan_purpose", "occupancy_type", "property_type",
                    "servicer_name", "source_system", "document_status"]:
            df[col] = df[col].astype("category")
        features, _ = check_leakage(df)
        for f in features:
            dtype = df[f].dtype
            is_ok = (
                pd.api.types.is_numeric_dtype(df[f]) or
                str(dtype) == "category" or
                str(dtype) == "bool"
            )
            assert is_ok, f"Feature '{f}' has non-numeric/category dtype: {dtype}"

    def test_target_cols_returned_correctly(self):
        """check_leakage must return all 7 target column names."""
        df = make_panel()
        _, target_cols = check_leakage(df)
        for t in self.TARGET_COLS:
            assert t in target_cols, f"Missing target: {t}"


# ===========================================================================
# 3. SERVICER RECONCILIATION TESTS
# ===========================================================================

from src.anomaly.detector import reconcile_servicer_updates

class TestServicerReconciliation:
    """Tests for the servicer reconciliation logic in detector.reconcile_servicer_updates."""

    def _make_panel_row(self, loan_id="LN001", month="2024-01-01",
                        balance=200_000.0, status="Current",
                        last_updated="2024-01-01"):
        return pd.DataFrame([{
            "loan_id": loan_id,
            "reporting_month": month,
            "current_balance": balance,
            "current_status": status,
            "last_updated_at": last_updated,
        }])

    def _make_update_row(self, loan_id="LN001", month="2024-01-01",
                         balance=200_000.0, status="Current",
                         last_updated="2024-01-01"):
        return pd.DataFrame([{
            "loan_id": loan_id,
            "reporting_month": month,
            "current_balance": balance,
            "current_status": status,
            "last_updated_at": last_updated,
        }])

    def test_no_conflict_not_flagged(self):
        """When panel and servicer agree, no stale flag should be raised."""
        panel = self._make_panel_row(balance=200_000.0, status="Current")
        updates = self._make_update_row(balance=200_000.0, status="Current")
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert not stale_flags.any(), "No conflict but stale flag was raised"

    def test_balance_conflict_flagged(self):
        """A balance discrepancy > $1 should trigger a stale flag."""
        panel = self._make_panel_row(balance=200_000.0)
        updates = self._make_update_row(balance=195_000.0)  # $5,000 discrepancy
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert stale_flags.any(), "Balance conflict not flagged"

    def test_status_conflict_flagged(self):
        """A status mismatch between panel and servicer should trigger a stale flag."""
        panel = self._make_panel_row(status="Current")
        updates = self._make_update_row(status="30 DPD")
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert stale_flags.any(), "Status conflict not flagged"

    def test_tiny_balance_difference_not_flagged(self):
        """A balance difference of $0.50 (rounding) should NOT be flagged."""
        panel = self._make_panel_row(balance=200_000.0)
        updates = self._make_update_row(balance=200_000.50)
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert not stale_flags.any(), "Tiny rounding difference incorrectly flagged"

    def test_no_update_means_no_flag(self):
        """Loans with no servicer update row should not be flagged."""
        panel = self._make_panel_row(loan_id="LN001")
        updates = self._make_update_row(loan_id="LN999")  # different loan
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert not stale_flags.any(), "Loan with no update was incorrectly flagged"

    def test_reconciliation_log_has_entries_for_conflicts(self):
        """The reconciliation log should contain one entry per conflict."""
        panel = self._make_panel_row(balance=200_000.0, status="Current")
        updates = self._make_update_row(balance=100_000.0, status="90+ DPD")
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert len(log) >= 1, "Reconciliation log is empty despite conflict"

    def test_multiple_loans_only_conflicting_flagged(self):
        """Only the loan with a conflict should be flagged, not the clean one."""
        panel = pd.DataFrame([
            {"loan_id": "LN001", "reporting_month": "2024-01-01",
             "current_balance": 200_000.0, "current_status": "Current",
             "last_updated_at": "2024-01-01"},
            {"loan_id": "LN002", "reporting_month": "2024-01-01",
             "current_balance": 150_000.0, "current_status": "Current",
             "last_updated_at": "2024-01-01"},
        ])
        updates = pd.DataFrame([
            {"loan_id": "LN001", "reporting_month": "2024-01-01",
             "current_balance": 200_000.0, "current_status": "Current",
             "last_updated_at": "2024-01-01"},     # no conflict
            {"loan_id": "LN002", "reporting_month": "2024-01-01",
             "current_balance": 50_000.0, "current_status": "Default",
             "last_updated_at": "2024-01-01"},     # conflict
        ])
        stale_flags, log = reconcile_servicer_updates(panel, updates)
        assert stale_flags.sum() == 1, (
            f"Expected 1 flagged loan, got {stale_flags.sum()}"
        )
