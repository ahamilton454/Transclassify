"""Layered transaction-string template engine.

Implements the composition model from data/transaction_patterns.md: a statement
line is a stack of layers (bank wrapper x processor prefix x merchant x locator x
ref tail), and structure-driven types (fees, payroll, ACH, P2P, ATM, government...)
are rendered from templates whose *form* is the category.

The LLM's only job (in generate.py) is to supply a category-appropriate merchant
and tag it with one of CHANNELS; `compose()` here turns that into a realistic string.
Everything is driven by a caller-supplied `random.Random` so generation is
reproducible and unit-testable — no module-level randomness.
"""
from __future__ import annotations

import random
import re

# Channels the merchant LLM may assign. compose() renders each differently.
CHANNELS = [
    "in_store",     # card-present retail / grocery / pharmacy
    "online",       # card-not-present e-commerce
    "subscription", # recurring SaaS / streaming / membership
    "gas",          # fuel pump
    "restaurant",   # dining (often Toast/Square)
    "ach_biller",   # recurring biller debit (utilities, insurance, rent, loan)
    "payroll",      # direct-deposit / payroll credit
    "p2p",          # Venmo / Zelle / Cash App / PayPal
    "atm_cash",     # ATM withdrawal
    "bank_fee",     # bank / processing fees
    "government",   # tax, DMV, municipal
    "check",        # paper check / mobile deposit
    "deposit",      # sales-revenue / card-batch deposit
]

# Channels whose amount lands as a credit (positive); everything else is an outflow.
CREDIT_CHANNELS = {"payroll", "deposit"}

CITIES = [
    ("PORTLAND", "OR"), ("AUSTIN", "TX"), ("DENVER", "CO"), ("SEATTLE", "WA"),
    ("CHICAGO", "IL"), ("BROOKLYN", "NY"), ("MIAMI", "FL"), ("ATLANTA", "GA"),
    ("PHOENIX", "AZ"), ("BOSTON", "MA"), ("NASHVILLE", "TN"), ("DALLAS", "TX"),
    ("SAN JOSE", "CA"), ("OAKLAND", "CA"), ("MINNEAPOLIS", "MN"), ("DETROIT", "MI"),
    ("COLUMBUS", "OH"), ("CHARLOTTE", "NC"), ("ORLANDO", "FL"), ("HOUSTON", "TX"),
    ("ST LOUIS", "MO"), ("PASADENA", "CA"), ("BOULDER", "CO"), ("RENO", "NV"),
    ("TUCSON", "AZ"), ("RALEIGH", "NC"), ("OMAHA", "NE"), ("FRESNO", "CA"),
]

STREETS = ["MAIN ST", "5TH AVE", "BROADWAY", "OAK ST", "MARKET ST", "1ST AVE",
           "ELM ST", "SUNSET BLVD", "PARK AVE", "WASHINGTON ST"]

# Account-holder names (for ACH INDN / P2P counterparties).
NAMES = ["JOHN SMITH", "MARIA GARCIA", "DAVID LEE", "SARAH JOHNSON", "MICHAEL BROWN",
         "EMILY DAVIS", "JAMES WILSON", "LINDA MARTINEZ", "ROBERT TAYLOR", "ANNA KIM"]

BANKS = ["WELLS FARGO", "CHASE", "BANK OF AMERICA", "PNC", "US BANK", "CITI"]

STAR_PREFIXES_RETAIL = ["SQ", "PP", "PYPL", "SUMUP", "IZ"]
STAR_PREFIXES_FOOD = ["TST", "SQ", "TOAST"]
ENTRY_DESCS = ["BILLPAY", "WEB PMT", "PAYMENT", "UTILITY", "INS PREM", "MORTGAGE",
               "ONLINE PMT", "ACH PMT", "RENT", "AUTOPAY"]
SEC_CODES = ["PPD", "WEB", "CCD"]
FEE_PHRASES = ["MONTHLY MAINTENANCE FEE", "OVERDRAFT FEE", "NSF FEE", "ATM FEE",
               "WIRE TRANSFER FEE", "INTEREST CHARGE ON PURCHASES",
               "FOREIGN TRANSACTION FEE", "RETURNED ITEM FEE", "SERVICE CHARGE",
               "CARD REPLACEMENT FEE", "EXCESS ACTIVITY FEE"]

# Amount ranges (USD magnitude) per channel — kept rough, just for realistic stored amounts.
AMOUNT_RANGES = {
    "in_store": (5, 220), "online": (8, 400), "subscription": (4, 80),
    "gas": (18, 95), "restaurant": (9, 160), "ach_biller": (40, 2400),
    "payroll": (900, 5200), "p2p": (10, 600), "atm_cash": (20, 400),
    "bank_fee": (3, 45), "government": (25, 3500), "check": (30, 2500),
    "deposit": (120, 6000),
}


# --------------------------------------------------------------------------- #
# Small field helpers
# --------------------------------------------------------------------------- #
def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _last4(rng: random.Random) -> str:
    return f"{rng.randint(0, 9999):04d}"


def _mmdd(rng: random.Random) -> str:
    return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"


def _mmdd_compact(rng: random.Random) -> str:
    return f"{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"


def _phone(rng: random.Random) -> str:
    area = rng.choice(["800", "888", "866", "877", "855", "402", "408", "650"])
    sep = rng.choice(["-", ""])
    return f"{area}{sep}{_digits(rng, 3)}{sep}{_digits(rng, 4)}"


def _city(rng: random.Random) -> tuple[str, str]:
    return rng.choice(CITIES)


def _city_st(rng: random.Random) -> str:
    city, st = _city(rng)
    return f"{city} {st}"


def merchant_field(name: str, rng: random.Random, cap: int = 22) -> str:
    """Uppercase, strip noise, and squeeze to a network char cap (Visa 25 / MC 22 / Sq 20)."""
    s = name.upper().strip()
    s = re.sub(r"[^A-Z0-9 &*'./#-]", "", s).strip()
    if len(s) > cap:
        squeezed = s.replace(" ", "")
        s = squeezed[:cap] if len(squeezed) > cap else squeezed
    return s.strip()


def _billing_url(merchant_uc: str, rng: random.Random) -> str:
    base = re.sub(r"[^A-Z0-9]", "", merchant_uc)[:12] or "PAY"
    return rng.choice([f"{base}.COM", f"WWW.{base}.COM", f"HELP.{base}.COM", f"{base}.COM/BILL"])


def _cnp_locator(merchant_uc: str, rng: random.Random) -> str:
    """Card-not-present locator: phone+ST or billing-URL+ST."""
    _, st = _city(rng)
    if rng.random() < 0.5:
        return f"{_phone(rng)} {st}"
    return f"{_billing_url(merchant_uc, rng)} {st}"


def _star_prefix(merchant_uc: str, rng: random.Random, pool: list[str]) -> str:
    p = rng.choice(pool)
    return rng.choice([f"{p} *{merchant_uc}", f"{p}*{merchant_uc}",
                       f"{p} * {merchant_uc}", f"{p}*  {merchant_uc}"])


# --------------------------------------------------------------------------- #
# Card wrappers (bank render styles)
# --------------------------------------------------------------------------- #
def _wrap_card(core: str, rng: random.Random, cnp: bool, merchant_uc: str,
               recurring: bool = False) -> str:
    loc = _cnp_locator(merchant_uc, rng) if cnp else _city_st(rng)
    style = rng.randint(0, 4)
    if style == 0:  # Wells Fargo
        verb = "RECURRING PAYMENT AUTHORIZED ON" if recurring else "PURCHASE AUTHORIZED ON"
        return f"{verb} {_mmdd(rng)} {core} {loc} S{_digits(rng, 15)} CARD {_last4(rng)}"
    if style == 1:  # Bank of America
        return f"CHECKCARD {_mmdd_compact(rng)} {core} {loc} {_digits(rng, 23)}"
    if style == 2:  # Chase (sometimes mixed-case)
        prefix = rng.choice(["", f"CARD PURCHASE {_mmdd(rng)} ", "RECURRING CARD PURCHASE " if recurring else ""])
        line = f"{prefix}{core} {loc}"
        return line.title() if rng.random() < 0.25 else line
    if style == 3:  # generic / credit union
        lead = rng.choice(["POS PURCHASE", "PURCHASE", "DEBIT CARD PURCHASE", "VISA PURCHASE", "POS DEBIT"])
        return f"{lead} {core} {loc}"
    # bare
    return f"{core} {loc}"


# --------------------------------------------------------------------------- #
# ACH render styles
# --------------------------------------------------------------------------- #
def _ach_line(company: str, rng: random.Random, entry: str, sec: str | None = None,
              credit: bool = False) -> str:
    co = company.upper()[:16].strip()
    name = rng.choice(NAMES)[:22]
    sec = sec or rng.choice(SEC_CODES)
    style = rng.randint(0, 4)
    if style == 0:  # BofA tagged
        return f"{co} DES:{entry} ID:{_digits(rng, 9)} INDN:{name} CO ID:{_digits(rng, 10)} {sec}"
    if style == 1:  # Chase labeled
        return (f"ORIG CO NAME:{co} ORIG ID:{_digits(rng, 10)} CO ENTRY DESCR:{entry} "
                f"SEC:{sec} TRACE#:{_digits(rng, 15)}")
    if style == 2:  # Wells Fargo positional
        return f"{co} {entry} {_digits(rng, 6)} {_digits(rng, 9)} {name}"
    if style == 3:  # generic prefix
        lead = "ACH CREDIT" if credit else rng.choice(["EFT DEBIT", "ACH DEBIT", "PREAUTHORIZED DEBIT"])
        return f"{lead} {co} {entry}"
    # online-bank prose
    verb = "ACH External Deposit from" if credit else "ACH External Withdrawal to"
    return f"{verb} {co.title()} ...{_last4(rng)}"


# --------------------------------------------------------------------------- #
# Channel builders
# --------------------------------------------------------------------------- #
def _in_store(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng)
    if rng.random() < 0.35:
        m = _star_prefix(m, rng, STAR_PREFIXES_RETAIL)
    if rng.random() < 0.3:
        m = f"{m} #{_digits(rng, 4)}"
    return _wrap_card(m, rng, cnp=False, merchant_uc=m)


def _online(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng)
    if rng.random() < 0.25:  # doubled name (merchant + its site)
        m = f"{m} {merchant_field(merchant, rng).replace(' ', '')}.COM"
    elif rng.random() < 0.3:
        m = _star_prefix(m, rng, ["PP", "PYPL", "PAYPAL", "AMZN Mktp US"])
    return _wrap_card(m, rng, cnp=True, merchant_uc=m, recurring=rng.random() < 0.15)


def _subscription(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng)
    style = rng.randint(0, 5)
    if style == 0:
        return f"APPLE.COM/BILL 866-712-7753 CA"
    if style == 1:
        return f"GOOGLE *{m} G.CO/HELPPAY# CA"
    if style == 2:
        return f"{m}.COM {_phone(rng)} {rng.choice([st for _, st in CITIES])}"
    if style == 3:
        return f"{rng.choice(['RECURRING', 'AUTOPAY', 'MONTHLY'])} {m}"
    if style == 4:
        return _wrap_card(m, rng, cnp=True, merchant_uc=m, recurring=True)
    return m


def _gas(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng, cap=18)
    city, st = _city(rng)
    style = rng.randint(0, 2)
    if style == 0:
        return f"{m} #{_digits(rng, 4)} {city} {st}"
    if style == 1:
        return f"{m} {_digits(rng, 5)} PUMP#{_digits(rng, 2)} {city} {st}"
    return f"{rng.choice(['POS PURCHASE', 'PURCHASE'])} {m} {city} {st}"


def _restaurant(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng)
    if rng.random() < 0.45:
        m = _star_prefix(m, rng, STAR_PREFIXES_FOOD)
    return _wrap_card(m, rng, cnp=False, merchant_uc=m)


def _ach_biller(company: str, rng: random.Random) -> str:
    return _ach_line(company, rng, entry=rng.choice(ENTRY_DESCS))


def _payroll(company: str, rng: random.Random) -> str:
    co = company.upper()[:16].strip()
    style = rng.randint(0, 3)
    if style == 0:
        return _ach_line(company, rng, entry="PAYROLL", sec="PPD", credit=True)
    if style == 1:
        return f"DIRECT DEPOSIT {co}"
    if style == 2:
        return f"{rng.choice(['ADP', 'GUSTO', 'PAYCHEX'])} *{co} PAYROLL"
    return f"{co} DES:PAYROLL ID:{_digits(rng, 9)} INDN:{rng.choice(NAMES)[:22]} PPD"


def _p2p(name: str, rng: random.Random) -> str:
    person = name if (name and " " in name and not name.isupper()) else rng.choice(NAMES)
    person = person.upper()
    app = rng.randint(0, 3)
    if app == 0:  # Venmo (generic — no handle on bank line)
        return rng.choice([f"VENMO PAYMENT {_digits(rng, 10)}", "VENMO CASHOUT",
                           f"VENMO *{person.split()[0]}"])
    if app == 1:  # Zelle (name shows)
        return rng.choice([f"ZELLE PAYMENT TO {person} {_digits(rng, 8)}",
                           f"ZELLE FROM {person}", f"ZELLE TO {person}"])
    if app == 2:  # Cash App
        return rng.choice([f"CASH APP*{person.split()[0]}", "CASH APP*CASH OUT"])
    return rng.choice([f"PAYPAL *{merchant_field(person, rng)}", f"PP*{merchant_field(person, rng, 16)}"])


def _atm_cash(_merchant: str, rng: random.Random) -> str:
    city, st = _city(rng)
    street = rng.choice(STREETS)
    if rng.random() < 0.4:
        return f"NON-{rng.choice(BANKS)} ATM WITHDRAWAL {_mmdd(rng)} {street} {city} {st}"
    return f"ATM WITHDRAWAL {_mmdd(rng)} #{_digits(rng, 6)} {street} {city} {st}"


def _bank_fee(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng) if merchant else ""
    if m and rng.random() < 0.45:
        return f"{m} {rng.choice(['PROCESSING FEE', 'FEE', 'SERVICE FEE'])}"
    return rng.choice(FEE_PHRASES)


def _government(agency: str, rng: random.Random) -> str:
    a = agency.upper()[:18].strip() if agency else "STATE"
    city, st = _city(rng)
    return rng.choice([
        f"IRS USATAXPYMT {_digits(rng, 12)}",
        f"IRS TREAS 310 TAX REF",
        f"{st} DTF TAX PAYMNT {_digits(rng, 9)}",
        f"{a} TAX PYMT {_digits(rng, 8)}",
        f"{st} DMV FEE {city}",
        f"CITY OF {city} {rng.choice(['UTILITY', 'PERMIT', 'WATER'])}",
        f"USPS PO {_digits(rng, 4)} {city} {st}",
    ])


def _check(_merchant: str, rng: random.Random) -> str:
    return rng.choice([
        f"CHECK # {_digits(rng, 4)}", "MOBILE DEPOSIT", "REMOTE ONLINE DEPOSIT",
        "MOBILE CHECK DEPOSIT", "COUNTER CREDIT", f"CHECK {_digits(rng, 4)}",
    ])


def _deposit(merchant: str, rng: random.Random) -> str:
    m = merchant_field(merchant, rng) if merchant else "SALES"
    proc = rng.choice(["SQ", "TST", "STRIPE", "TOAST", "CLOVER", "DOORDASH", "PAYPAL"])
    return rng.choice([
        f"{proc} DES:DEPOSIT INDN:{m} CO ID:{_digits(rng, 10)} CCD",
        f"{proc} *{m} DEPOSIT",
        f"DEPOSIT {proc} {m}",
        f"{m} DES:SALES ID:{_digits(rng, 9)} CCD",
    ])


_BUILDERS = {
    "in_store": _in_store, "online": _online, "subscription": _subscription,
    "gas": _gas, "restaurant": _restaurant, "ach_biller": _ach_biller,
    "payroll": _payroll, "p2p": _p2p, "atm_cash": _atm_cash, "bank_fee": _bank_fee,
    "government": _government, "check": _check, "deposit": _deposit,
}


def compose(merchant: str, channel: str, rng: random.Random) -> str:
    """Render one realistic statement line for (merchant, channel)."""
    builder = _BUILDERS.get(channel, _in_store)
    line = builder(merchant, rng)
    return re.sub(r"\s+", " ", line).strip()


def amount_for(channel: str, rng: random.Random) -> float:
    """A plausible signed amount: credits positive, everything else negative."""
    lo, hi = AMOUNT_RANGES.get(channel, (5, 200))
    mag = round(rng.uniform(lo, hi), 2)
    return mag if channel in CREDIT_CHANNELS else -mag
