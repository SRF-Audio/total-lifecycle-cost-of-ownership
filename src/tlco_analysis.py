from __future__ import annotations
import math, logging
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger("TLCO")

def setup_logging(log_path: str) -> None:
    """Initialize console + file logging once per run."""
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(ch); logger.addHandler(fh)

def get_value(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data[key]
    logger.warning("Missing key '%s'; using default=%s", key, default)
    return default

def clamp(value: float, min_v: float, max_v: float) -> float:
    if value < min_v:
        logger.warning("Value %.4f clamped to min %.4f", value, min_v)
        return min_v
    if value > max_v:
        logger.warning("Value %.4f clamped to max %.4f", value, max_v)
        return max_v
    return value

def annualize_rate(total_rate: float, years: float) -> float:
    """Convert total depreciation fraction over 'years' to annualized compound rate (negative)."""
    try:
        total_rate = clamp(total_rate, 0.0, 0.999999)
        remain = 1 - total_rate
        if years <= 0:
            raise ValueError("years must be > 0")
        return (remain ** (1 / years)) - 1
    except Exception as e:
        logger.error("annualize_rate failed: %s", e)
        return -total_rate / max(years, 1.0)

def infer_annual_depreciation_rate(purchase_price: float, quotes: Optional[List[Dict[str, Any]]], default_rate_by_segment: float) -> float:
    """
    Estimate annual depreciation from provided quotes (msrp/current_resale/model_year/age_years),
    otherwise fall back to a segment default. Returns negative fraction per year.
    """
    try:
        if quotes:
            best, best_model_year = None, -1
            for q in quotes:
                if "msrp" in q and "current_resale" in q and "model_year" in q:
                    if q["model_year"] > best_model_year:
                        best, best_model_year = q, q["model_year"]
            if best:
                msrp = float(best["msrp"]); resale = float(best["current_resale"]); model_year = int(best["model_year"])
                age_years = max(1.0, float(get_value(best, "age_years", 2025 - model_year)))
                if resale <= 0 or msrp <= 0:
                    raise ValueError("Invalid quote numbers.")
                total_dep_fraction = clamp((msrp - resale) / msrp, 0.0, 0.999999)
                rate = annualize_rate(total_dep_fraction, age_years)
                logger.info("Inferred depreciation from %s: %.2f%%/yr", model_year, rate * 100.0)
                return rate
        logger.info("Using default segment depreciation rate: %.2f%%/yr", default_rate_by_segment * 100)
        return -abs(default_rate_by_segment)
    except Exception as e:
        logger.error("infer_annual_depreciation_rate failed: %s", e); return -abs(default_rate_by_segment)

def project_resale_value(purchase_price: float, annual_dep_rate: float, years: int) -> float:
    try:
        years = max(0, int(years))
        factor = (1.0 + annual_dep_rate) ** years
        return max(0.0, purchase_price * factor)
    except Exception as e:
        logger.error("project_resale_value failed: %s", e); return 0.0

def compute_annual_fuel_cost(annual_miles: float, fuel_type: str, mpg: Optional[float], mpge: Optional[float], kwh_per_100_miles: Optional[float], gas_price_per_gal: float, electricity_price_per_kwh: float, phev_utilization_electric_fraction: float = 0.6) -> float:
    """Annual energy spend for gas/hybrid/ev/phev."""
    try:
        fuel_type = (fuel_type or "").lower()
        miles = max(0.0, float(annual_miles))
        gas_price = max(0.0, float(gas_price_per_gal))
        elec_price = max(0.0, float(electricity_price_per_kwh))

        if fuel_type in ("gas", "hybrid"):
            if mpg is None or mpg <= 0:
                raise ValueError("mpg must be provided for gas/hybrid.")
            return (miles / mpg) * gas_price

        if fuel_type == "ev":
            if kwh_per_100_miles and kwh_per_100_miles > 0:
                kwh_100 = kwh_per_100_miles
            elif mpge and mpge > 0:
                kwh_100 = 33.7 * 100.0 / mpge
            else:
                raise ValueError("Provide kwh_per_100_miles or mpge for EV.")
            return ((miles / 100.0) * kwh_100) * elec_price

        if fuel_type == "phev":
            ef = clamp(float(phev_utilization_electric_fraction), 0.0, 1.0)
            if kwh_per_100_miles and kwh_per_100_miles > 0:
                kwh_100 = kwh_per_100_miles
            elif mpge and mpge > 0:
                kwh_100 = 33.7 * 100.0 / mpge
            else:
                kwh_100 = 40.0
                logger.warning("Missing PHEV electric efficiency; assuming %.1f kWh/100mi", kwh_100)
            elec_cost = ((miles * ef / 100.0) * kwh_100) * elec_price
            if mpg is None or mpg <= 0:
                mpg = 35.0
                logger.warning("Missing PHEV mpg; assuming %.1f mpg for gas portion", mpg)
            gas_cost = ((miles * (1 - ef)) / mpg) * gas_price
            return elec_cost + gas_cost

        raise ValueError(f"Unsupported fuel_type={fuel_type}")
    except Exception as e:
        logger.error("compute_annual_fuel_cost failed: %s", e)
        return 0.0

def compute_insurance_cost(annual_premium: float) -> float:
    try:
        return max(0.0, float(annual_premium))
    except Exception as e:
        logger.error("compute_insurance_cost failed: %s", e)
        return 0.0

def compute_annual_maintenance_cost(year_index: int, maintenance_schedule: List[Dict[str, Any]], annual_miles: float) -> float:
    """
    Trigger tasks by miles or years for this specific year.
    A task fires if:
      - year_index is a multiple of interval_years, or
      - cumulative miles crossed a new multiple of interval_miles this year.
    """
    try:
        total = 0.0
        cum_miles = annual_miles * year_index
        for item in maintenance_schedule or []:
            cost = float(get_value(item, "cost", 0.0))
            im = float(get_value(item, "interval_miles", 0.0) or 0.0)
            iy = float(get_value(item, "interval_years", 0.0) or 0.0)
            triggered = False
            if iy and iy > 0 and (year_index % int(iy) == 0):
                triggered = True
            if im and im > 0 and cum_miles > 0:
                prev = int(annual_miles * (year_index - 1))
                if int(cum_miles) // int(im) == (prev // int(im)) + 1:
                    triggered = True
            if triggered:
                total += cost
        return total
    except Exception as e:
        logger.error("compute_annual_maintenance_cost failed: %s", e)
        return 0.0

def compute_annual_consumables_cost(year_index: int, consumables: Dict[str, Any], annual_miles: float) -> float:
    """Estimate consumables (tires, pads, wipers, 12V battery)."""
    try:
        total = 0.0
        cum_miles = annual_miles * year_index

        # Tires
        tires = get_value(consumables or {}, "tires", {})
        if tires:
            miles_per_set = float(get_value(tires, "miles_per_set", 40000))
            cost_per_set = float(get_value(tires, "cost_per_set", 800))
            if miles_per_set > 0:
                prev_sets = math.floor((annual_miles * (year_index - 1)) / miles_per_set)
                curr_sets = math.floor(cum_miles / miles_per_set)
                sets_this_year = max(0, curr_sets - prev_sets)
                total += sets_this_year * cost_per_set

        # Brake pads
        pads = get_value(consumables or {}, "brake_pads", {})
        if pads:
            miles_per_set = float(get_value(pads, "miles_per_set", 35000))
            cost_per_axle = float(get_value(pads, "cost_per_axle", 225))
            axles = int(get_value(pads, "axles", 2))
            if miles_per_set > 0:
                prev_sets = math.floor((annual_miles * (year_index - 1)) / miles_per_set)
                curr_sets = math.floor(cum_miles / miles_per_set)
                sets_this_year = max(0, curr_sets - prev_sets)
                total += sets_this_year * cost_per_axle * axles

        # Wipers
        wipers = get_value(consumables or {}, "wiper_blades", {})
        if wipers:
            replacements_per_year = float(get_value(wipers, "replacements_per_year", 1))
            cost_per_set = float(get_value(wipers, "cost_per_set", 35))
            total += replacements_per_year * cost_per_set

        # 12V battery
        aux_batt = get_value(consumables or {}, "aux_12v_battery", {})
        if aux_batt:
            interval_years = float(get_value(aux_batt, "interval_years", 4))
            cost = float(get_value(aux_batt, "cost", 200))
            if interval_years > 0 and year_index % int(interval_years) == 0:
                total += cost

        return total
    except Exception as e:
        logger.error("compute_annual_consumables_cost failed: %s", e)
        return 0.0

def compute_purchase_taxes_and_fees(purchase_price: float, tax_rate: float, doc_and_title_fees: float) -> float:
    try:
        tax = max(0.0, purchase_price) * max(0.0, tax_rate)
        return tax + max(0.0, doc_and_title_fees)
    except Exception as e:
        logger.error("compute_purchase_taxes_and_fees failed: %s", e)
        return 0.0

def compute_annual_registration_fee(base_fee: float) -> float:
    try:
        return max(0.0, float(base_fee))
    except Exception as e:
        logger.error("compute_annual_registration_fee failed: %s", e)
        return 0.0

def compute_reliability_risk_premium(reliability_score_out_of_5: Optional[float], safety_rating_out_of_5: Optional[float], annual_miles: float, years: int, base_unexpected_repairs_per_year: float = 150.0) -> float:
    """
    Convert reliability & safety into a contingency budget (lower when scores are higher).
    Returns a total $ contingency over the horizon.
    """
    try:
        def c(v, a, b): return max(a, min(b, v))
        reliability = c(float(reliability_score_out_of_5 or 3.0), 1.0, 5.0)
        safety = c(float(safety_rating_out_of_5 or 4.0), 1.0, 5.0)
        rel_multiplier = 2.0 - (reliability - 1.0) * (1.4 / 4.0)   # ~[2.0 .. 0.6]
        safety_multiplier = 1.0 - (safety - 3.0) * (0.06 / 2.0)    # +/- ~6%
        return max(0.0, base_unexpected_repairs_per_year * years * rel_multiplier * safety_multiplier)
    except Exception as e:
        logger.error("compute_reliability_risk_premium failed: %s", e)
        return 0.0

def compute_financing_cost(amount_financed: float, apr: float, term_years: int) -> float:
    """Simple interest approximation (conservative)."""
    try:
        principal = max(0.0, float(amount_financed))
        apr = max(0.0, float(apr))
        years = max(0, int(term_years))
        return principal * apr * years
    except Exception as e:
        logger.error("compute_financing_cost failed: %s", e)
        return 0.0

def npv(cash_flows_by_year: List[float], discount_rate: float) -> float:
    try:
        r = max(0.0, float(discount_rate))
        total = 0.0
        for t, cf in enumerate(cash_flows_by_year, start=1):
            total += cf / ((1 + r) ** t)
        return total
    except Exception as e:
        logger.error("npv failed: %s", e)
        return sum(cash_flows_by_year or [])

def compute_tlco_for_option(option: Dict[str, Any], horizon_years: int) -> Dict[str, Any]:
    """Compute TLCO components for one vehicle option."""
    try:
        year = int(get_value(option, "year", 0) or 0)
        make = get_value(option, "make", "Unknown")
        model = get_value(option, "model", "Unknown")
        trim = get_value(option, "trim", "Unknown")
        identifier = f"{year} {make} {model} {trim}"

        purchase_price = float(get_value(option, "purchase_price", 0.0) or 0.0)
        tax_rate = float(get_value(option, "tax_rate", 0.0))
        doc_and_title_fees = float(get_value(option, "doc_and_title_fees", 300.0))
        annual_miles = float(get_value(option, "annual_miles", 12000.0))

        # Fuel/energy
        fuel_type = get_value(option, "fuel_type", "hybrid")
        mpg = option.get("mpg")
        mpge = option.get("mpge")
        kwh_100 = option.get("kwh_per_100_miles")
        gas_price = float(get_value(option, "gas_price_per_gal", 3.50))
        elec_price = float(get_value(option, "electricity_cost_per_kwh", 0.14))
        phev_e_frac = float(get_value(option, "phev_utilization_electric_fraction", 0.6))

        # Insurance/registration
        insurance_per_year = float(get_value(option, "insurance_annual_premium", 1400.0))
        registration_base_fee = float(get_value(option, "registration_annual_fee", 180.0))

        # Maintenance & consumables
        maintenance_schedule = get_value(option, "maintenance_schedule", [])
        consumables = get_value(option, "consumables", {})

        # Reliability & safety
        reliability_score = option.get("reliability_score_out_of_5")
        safety_rating = option.get("safety_rating_out_of_5")

        # Depreciation
        segment_default_dep = float(get_value(option, "segment_default_dep_rate", 0.12))
        prior_quotes = option.get("depreciation_quotes")
        annual_dep_rate = infer_annual_depreciation_rate(purchase_price, prior_quotes, segment_default_dep)
        resale_after_horizon = project_resale_value(purchase_price, annual_dep_rate, horizon_years)

        # Up-front taxes & fees
        upfront = compute_purchase_taxes_and_fees(purchase_price, tax_rate, doc_and_title_fees)

        # Yearly flows
        fuel_costs, insurance_costs, maintenance_costs, consumable_costs, registration_costs = [], [], [], [], []
        for y in range(1, horizon_years + 1):
            fuel = compute_annual_fuel_cost(annual_miles, fuel_type, mpg, mpge, kwh_100, gas_price, elec_price, phev_e_frac)
            insurance = compute_insurance_cost(insurance_per_year)
            maintenance = compute_annual_maintenance_cost(y, maintenance_schedule, annual_miles)
            consumable = compute_annual_consumables_cost(y, consumables, annual_miles)
            registration = compute_annual_registration_fee(registration_base_fee)

            fuel_costs.append(fuel)
            insurance_costs.append(insurance)
            maintenance_costs.append(maintenance)
            consumable_costs.append(consumable)
            registration_costs.append(registration)

        # Reliability contingency
        reliability_contingency = compute_reliability_risk_premium(reliability_score, safety_rating, annual_miles, horizon_years)

        # Financing (optional)
        financed_amount = float(get_value(option, "amount_financed", 0.0))
        apr = float(get_value(option, "apr", 0.0))
        term_years = int(get_value(option, "loan_term_years", 0) or 0)
        financing_total_interest = compute_financing_cost(financed_amount, apr, term_years)

        # Sum
        total_operating = sum(fuel_costs) + sum(insurance_costs) + sum(maintenance_costs) + sum(consumable_costs) + sum(registration_costs)
        total_cost = purchase_price + upfront + total_operating + reliability_contingency + financing_total_interest - resale_after_horizon

        # Optional NPV
        discount_rate = float(get_value(option, "discount_rate", 0.0))
        if discount_rate > 0:
            operating_npv = npv(
                [fuel_costs[i] + insurance_costs[i] + maintenance_costs[i] + consumable_costs[i] + registration_costs[i] for i in range(horizon_years)],
                discount_rate,
            )
        else:
            operating_npv = None

        return {
            "id": identifier,
            "year": year, "make": make, "model": model, "trim": trim,
            "purchase_price": round(purchase_price, 2),
            "upfront_tax_and_fees": round(upfront, 2),
            "annual_dep_rate_pct": round(annual_dep_rate * 100.0, 2),
            "projected_resale_value": round(resale_after_horizon, 2),
            "fuel_total": round(sum(fuel_costs), 2),
            "insurance_total": round(sum(insurance_costs), 2),
            "maintenance_total": round(sum(maintenance_costs), 2),
            "consumables_total": round(sum(consumable_costs), 2),
            "registration_total": round(sum(registration_costs), 2),
            "reliability_contingency": round(reliability_contingency, 2),
            "financing_interest_total": round(financing_total_interest, 2),
            "operating_total": round(total_operating, 2),
            "tlco_total_cash": round(total_cost, 2),
            "operating_npv_at_discount_rate": round(operating_npv, 2) if operating_npv is not None else None,
        }
    except Exception as e:
        logger.exception("compute_tlco_for_option failed for option=%s", option)
        return {"error": str(e), "id": option.get("trim", "unknown")}

def compute_tlco(input_json: Dict[str, Any], log_path: str) -> pd.DataFrame:
    """Compute TLCO for all options and return a DataFrame."""
    setup_logging(log_path)
    try:
        horizon_years = int(get_value(input_json, "horizon_years", 7))
        options = get_value(input_json, "options", [])
        results: List[Dict[str, Any]] = []
        for opt in options:
            results.append(compute_tlco_for_option(opt, horizon_years))
        df = pd.DataFrame(results)
        ordering = [
            "id","purchase_price","upfront_tax_and_fees",
            "annual_dep_rate_pct","projected_resale_value",
            "fuel_total","insurance_total","maintenance_total",
            "consumables_total","registration_total",
            "reliability_contingency","financing_interest_total",
            "operating_total","tlco_total_cash",
            "operating_npv_at_discount_rate"
        ]
        cols = [c for c in ordering if c in df.columns] + [c for c in df.columns if c not in ordering]
        return df[cols]
    except Exception as e:
        logger.exception("compute_tlco failed")
        return pd.DataFrame([{"error": str(e)}])
