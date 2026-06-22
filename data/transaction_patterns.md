# Transaction-string pattern catalog

Synthesized from deep research (sources at bottom) to drive a realistic transaction generator.
**The core model: a statement line is a stack of independent layers** — render each separately and
compose, instead of asking an LLM for a whole "messy string."

```
[bank wrapper] [rail/processor prefix *] MERCHANT/DBA [#store] [LOCATOR] [REF/IDENTIFIER TAIL]
```

Example assembled from layers:
`PURCHASE AUTHORIZED ON 06/14 SQ *PERK UP COFFEE PORTLAND OR S304881225500031 CARD 7788`
= WF wrapper + Square prefix + truncated merchant + city/state + auth ref + card last-4.

Hard constraints (encode as generator rules):
- Merchant/DBA field **≤ ~22 chars** (Visa 25 / Mastercard 22 / **Square 20**); city field **≤ 13**; state = 2 letters; intl appends a 3-letter ISO country code (GBR/FRA/…).
- `*` is the **universal separator** between an aggregator prefix and the sub-merchant/product.
- Most lines are **ALL CAPS** (issuer uppercases); Chase mixes case; double-spaces appear where a field is blank.
- Truncation is mid-word, vowels/spaces/punctuation dropped (`WHOLEFDS BWY 10455`, `PP*AMAZONMARK`).

---

## Layer 1 — Bank render-style / wrapper (pick ONE per line)

| Style | Card template | ACH template |
|---|---|---|
| **Chase** | `POS DEBIT {desc}` · `CARD PURCHASE {MM/DD} {desc}` · `RECURRING CARD PURCHASE {desc}` | `ORIG CO NAME:{co} ORIG ID:{id} DESC DATE:{YYMMDD} CO ENTRY DESCR:{ed} SEC:{sec} TRACE#:{15} IND ID:{id} IND NAME:{name}` |
| **Wells Fargo** | `PURCHASE AUTHORIZED ON {MM/DD} {desc} {CITY} {ST} S{15} CARD {last4}` (RECURRING → `RECURRING PAYMENT AUTHORIZED ON …`; PIN → `P` ref) | positional, unlabeled: `{co} {entry_desc} {YYMMDD} {ind_id} {NAME}` |
| **Bank of America** | `CHECKCARD {MMDD} {desc} {CITY} {ST} {ARN}` · `POS PURCHASE {MMDD} {desc}` | tagged: `{co} DES:{ed} ID:{id} INDN:{NAME} CO ID:{co_id} {SEC}` |
| **Generic / CU / online bank** | `PURCHASE {desc}` · `POS PURCHASE {desc}` · `DEBIT CARD PURCHASE {desc}` · `VISA PURCHASE {desc}` | `EFT DEBIT {co}` · `ACH DEBIT {co}` · `PREAUTHORIZED DEBIT {co}` · `ELECTRONIC WITHDRAWAL {co}` · Cap One/Ally prose `ACH External Transfer to {bank} ...{last4}` |

The wrapper decides whether the SEC code is `SEC:WEB`, trailing bare ` PPD`, or invisible, and whether
ACH fields are labeled (`DES:/INDN:/CO ID:`) or positional.

## Layer 2 — Rail / processor prefix

| Entity | Prefix | Const phone | Notes |
|---|---|---|---|
| Square | `SQ *` | — | ≤20 after prefix; optional `*#store`, `gosq.com` |
| Toast (restaurants) | `TST*` / `TST* ` | — | merchant always a restaurant/bar |
| Stripe | `{DBA}* {suffix}` | — | suffix = order#/date/plan; bare `STRIPE`/`STRIPE PAYMENTS` if unconfigured |
| PayPal | `PAYPAL *` / `PP*` / `PYPL*` | `402-935-7733` (+CA) | hard-compresses names; `INST XFER` = transfer |
| Braintree | `{co}*{product} {phone} {ST}` | — | Uber `UBER *EATS 8005928996 CA` |
| Cash App / Block | `CASH APP*` / `SQ *` / `SQC*` / `CK*` | `877-417-4551` | name or `CASH OUT`; `CK*MERCHANT` = Cash Card spend |
| Venmo | `VENMO` / `VENMO*` | `855-812-4430` (+NY) | generic, **no name/@handle**; `PAYMENT`/`CASHOUT`/`INSTANT` |
| Google billing | `GOOGLE *{App}` | — | one line **per app**; `g.co/helppay#` + `CA` |
| Apple billing | `APPLE.COM/BILL` / `ITUNES.COM/BILL` / `APL*` | `866-712-7753` | **one consolidated line**, no service name |
| Microsoft | `MSFT *<E0…> MSBILL.INFO` / `MICROSOFT*{prod}` | — | angle-bracket code is real |
| Amazon | `AMZN Mktp US*{id}` / `Amazon.com*{id}` / `AMZN Digital*` / `Amazon Prime*` / `Prime Video *` | `888-802-3080` (Prime) | random 6–9-char order-id; one order → many lines; `AMZN.COM/BILL WA` |
| Steam | `STEAMGAMES.COM` | `425-952-2985` | region `WA`/`GB` |
| PlayStation | `PLAYSTATIONNETWORK` / `SONY *` | `800-345-7669` | Foster City CA |
| SumUp / Zettle / Clover | `SUMUP *` / `Zettle *`/`IZ*` / (Clover: bare DBA) | — | small-merchant POS |

Apple Pay / Google Pay / Samsung Pay **tap-to-pay add NO prefix** — they pass through the underlying
merchant descriptor.

## Layer 3 — Locator (after the merchant)

- Card-present: `{CITY≤13} {ST}` (+ `US`/country for intl) — `OAKLAND CA`, `LONDON GBR`.
- Card-not-present: `{PHONE} {ST}` (`866-579-7172 CA`) **or** `{BILLING_URL} {ST}` (`AMZN.COM/BILL WA`, `G.CO/HELPPAY CA`, `MSBILL.INFO`).
- Gas: `… {BRAND} {store#} PUMP#{nn} {CITY} {ST}` (+ separate $1/$75–175 pre-auth hold).

## Layer 4 — Reference / identifier tail

- Card: `S{15} CARD {last4}` (WF) · 23-digit ARN (BofA) · `CARD {last4}`.
- Store ids: `#1234`, Target `T-2710`, McD `F1234`, bare 5-digit (gas).
- ACH (NACHA field caps): company 16 · entry-desc 10 · individual name 22 · company-id 10 · individual-id 15 · trace 15 (8-routing+7-seq). Labels: `PPD ID:` / `CO ID:` / `WEB ID:` / `ORIG ID:` / `IND ID:` / `TRACE#:`. Real CO IDs: Progressive `9409348021`, GEICO `3530075853`, AT&T `9864031004`, Gusto `9138864003`, ADP `9659605001/2`.

## Layer 5 — Realism mutators (randomize)

case (UPPER vs Mixed) · `*`-spacing jitter (`SQ *X`/`SQ* X`/`SQ*X`) · truncate to 15–22 mid-word ·
random order-ids (Amazon 6–9 alnum; Spotify `P`+10 digits; MSFT `<E0…>`) · doubled merchant name on a
minority of CNP lines (`WALMART WALMART.COM 800-925-6278 AR`) · pending vs posted (pending strips the
ref, sometimes shows the *processor* name) · append const phone/help-URL to ~30–50% of CNP lines.

---

## Structure-driven types (the descriptor *is* the category, regardless of merchant)

These should be generated from templates, not merchant names — the label comes from the structure:

- **Fees / Bank charges:** `MONTHLY MAINTENANCE FEE`, `OVERDRAFT FEE`, `NSF FEE`, `INTEREST CHARGE ON PURCHASES`, `FOREIGN TRANSACTION FEE`, `ATM SURCHARGE FEE 3.00 CARDTRONICS …`, processor fees `ELAVON BATCH FEE`, `STRIPE PROCESSING FEE`.
- **Transfers / P2P:** `VENMO CASHOUT`, `ZELLE PAYMENT {name}` (bank-templated, name shows), `CASH APP*CASH OUT`, `Online Transfer To Chk ...6276`, `TRANSFER FROM SAVINGS`, `WIRE TRANSFER IN`.
- **Income / payroll (ACH credit, PPD):** `ACME CORP DES:PAYROLL …`, `DIRECT DEPOSIT ACME CORP`, `ADP * ACME CORP PAYROLL`, `IRS TREAS 310 TAX REF` (refund), `Zelle Payment From {name}`.
- **Taxes & government:** `IRS USATAXPYMT {ref}`, `IRS TREAS 310 TAX REF`, `CA FTB WEB PAY`, `FRANCHISE TAX BO PAYMENTS`, `NYS DTF PIT TAX PAYMNT`, `CA DMV VEH REG FEE`, `SECRETARY OF STATE CORP FILG FEE`, `USPS PO 0561 BROOKLYN NY`.
- **Returns / reversals / refunds:** `PURCHASE RETURN {merchant}`, `POS REFUND {merchant}`, `REVERSAL`, `MISC CREDIT`, ACH returns `ACH RETURN R01 INSUFFICIENT FUNDS {co}` (weight R01/R02/R03/R10/R29; ~half of consumer banks drop the code → bare `NSF`/`RETURNED`).
- **ATM / cash:** `ATM WITHDRAWAL 06/14 #000234 5TH AVE NEW YORK NY`, `NON-WF ATM WITHDRAWAL …`.
- **Checks / deposits:** `CHECK # 1234`, `MOBILE DEPOSIT`, `REMOTE ONLINE DEPOSIT`, `COUNTER CREDIT`.
- **Subscriptions:** native descriptor (`NETFLIX.COM`, `SPOTIFY USA`, `ADOBE *CREATIVE CLOUD`, `OPENAI *CHATGPT`, `MSFT *<E0…> MSBILL.INFO`) **OR app-store collapse** — a real share render as `APPLE.COM/BILL` / `GOOGLE *{App}`. `RECURRING`/`AUTOPAY`/`MONTHLY` may prefix/suffix.

---

## How the generator should use this (hybrid: templates × LLM merchant)

The LLM is great at **real merchant names**; bad at **realistic format diversity**. So split the job:

1. **Pick the category** (target gold).
2. **Decide merchant-driven vs structure-driven:**
   - *Merchant-driven* (Groceries, Gas, Restaurants, Software, Shopping…): the **LLM supplies a
     category-appropriate merchant** (+ optional sub-product); a **code template engine** wraps it in a
     compatible rail/wrapper/locator/ref and applies mutators. Gold = the category (the merchant
     determines it).
   - *Structure-driven* (Fees, Transfers, Income/Payroll, Taxes, Returns, ATM): generate from the
     **structural template directly** — the descriptor's form *is* the category. Gold = that category.
3. **Respect compatibility:** map each category/merchant-type → plausible rails (e.g. gas → card-present
   POS + `PUMP#`; SaaS sub → CNP card with billing-URL *or* app-store collapse; payroll → ACH PPD credit;
   insurance autopay → ACH `… INS PREM PPD ID:`). Never wrap a payroll deposit in `PUMP#07`.
4. **Apply Layer-5 mutators** for surface noise.

This yields realistic strings **and** trustworthy gold (the merchant/structure fixes the label, not the
LLM's opinion), and far more format diversity than asking the LLM for "messy" strings.

---

## Sources
Modern Treasury (decode statements, SEC codes, ACH return codes), NACHA rules (PAYROLL/PURCHASE entry
descriptions, file fields), Visa Merchant Data Standards Manual, Stripe/Adyen/Braintree/Recurly/
Chargebee descriptor docs, Square/Toast/Clover/SumUp/Zettle help, Cash App/Venmo/Zelle/PayPal/Apple Cash
descriptor pages, IRS TREAS 310 / USATAXPYMT, CA FTB Web Pay, ParkMobile PMUSA, Wells Fargo "read your
statement", per-merchant charge-identifier pages (Apple/Google/Microsoft/Amazon/Steam/PlayStation/Adobe/
Spotify/Netflix/ABC Financial). Exact bank wrapper ref-digit counts are approximate (banks don't publish
verbatim grammars); the char caps, `*` conventions, SEC codes, and merchant prefixes/phones are spec-confirmed.
