# Derived fields — what was inferred and how confident I am

The web form asks 14 questions. Three model inputs are **computed from those
answers on the server**, never asked. Two of them (`income_to_poverty_ratio`,
`food_desert_flag`) are produced by a generator whose source code we do not
have, so their exact rules were **recovered empirically** from the 100k-row
dataset. This file records what each rule is, how it was recovered, and how
much to trust it.

The recovery script lives in `train.py` (`recover_and_report_rules()`), so the
numbers below regenerate on every training run.

---

## 1. `rent_or_mortgage_burden_pct` — confidence: exact

Defined outright in the data dictionary: *"Monthly housing cost divided by
monthly income, capped at 200 percent."*

```
rent_or_mortgage_burden_pct = min(100 * monthly_housing_cost_usd / monthly_income_usd, 200)
```

No inference needed. Income is clamped to a floor of 1 before dividing so a
`$0` income entry cannot divide-by-zero (the form enforces a $300 minimum
anyway).

---

## 2. `income_to_poverty_ratio` — confidence: high

The dictionary calls this *"income-to-poverty ratio using an approximate
guideline-style denominator."* I recovered the denominator by solving
`denominator = (monthly_income * 12) / income_to_poverty_ratio` for every row
and grouping by `household_size`:

| household_size | implied annual denominator (median) |
|---:|---:|
| 1 | 15,060 |
| 2 | 20,440 |
| 3 | 25,820 |
| 4 | 31,200 |
| 5 | 36,583 |
| 6 | 41,958 |
| 7 | 47,330 |
| 8 | 52,724 |

The step between consecutive sizes is a near-constant **$5,380**, and size 1 is
**$15,060**. Those are the **2024 HHS Federal Poverty Guidelines** for the 48
contiguous states ($15,060 for one person, +$5,380 per additional person). So:

```
denominator          = 15060 + 5380 * (household_size - 1)
income_to_poverty_ratio = (monthly_income_usd * 12) / denominator
```

Recovering the ratio this way reproduces the dataset column with a **median
absolute error of 0.0025** (99th percentile 0.0074) — the small residual is
noise the generator added on top of the clean guideline formula. I use the
clean formula in the app.

**Why high and not exact:** the guideline match is unmistakable, but the
generator adds a little jitter I do not replicate, and I assume the "48
contiguous states" schedule (Alaska/Hawaii have different guidelines). For a
teaching app that is immaterial.

---

## 3. `food_desert_flag` — confidence: high on the true rule, medium on the
form-only approximation the app uses

**The true generator rule** (recovered by cross-tabulating
`distance_to_grocery_miles` and `transportation_access` — and then
`urbanicity` — against the flag):

- If `transportation_access == "reliable"` → **never** a food desert
  (flag rate 0.0 across all 80,665 such rows, at every distance).
- Otherwise the flag turns on above a distance threshold that **depends on
  urbanicity**:
  - urban / suburban: flag = 1 when distance ≳ **1.4 mi** (last flag-0 at 1.3, first flag-1 at 1.4)
  - rural: flag = 1 when distance ≳ **8.5 mi** (last flag-0 at 8.5, first flag-1 at 8.6)

  This mirrors the USDA "low-income low-access" convention of a tighter
  distance cutoff in denser areas and a looser one in rural areas.

**The problem:** `urbanicity` is **not one of the 14 form fields** (by design —
see below). The prompt asks the app to derive the flag from
`distance_to_grocery_miles` and `transportation_access` *only*. So the app uses
a single-threshold approximation:

```
food_desert_flag = 1  if  transportation_access != "reliable"  and  distance_to_grocery_miles > 1.5
                   0  otherwise
```

**Agreement with the true flag: 97.1%** across all 100k rows (best single
threshold on a 0.5-mi grid). It matches the true rule almost perfectly for
urban/suburban households (the majority) and over-flags some rural households
who live 1.5–8.5 mi out with unreliable transport — arguably a defensible
error for a screening tool, since those households genuinely have constrained
food access.

**Why medium:** the 1.5-mi cutoff is an approximation forced by not collecting
urbanicity, and it is deliberately conservative (flags more, not fewer). If a
future version adds an urbanicity question, switch to the two-threshold rule
above and agreement goes to ~100%.

---

## Note on protected attributes

`urbanicity` is dropped from the form for simplicity, but the four **fairness-
audit dimensions** — `race_ethnicity`, `immigrant_household`,
`primary_language`, `disability_present` — are dropped **on purpose**. The app
never asks for them and never conditions predictions on them. They exist only
for the offline fairness audit. This is stated on the About page.
