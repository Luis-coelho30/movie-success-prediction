# Movie Success Prediction with Decision Trees

**Authors**
- Luís Augusto Coelho de Souza (RA00331675)
- Guilherme Schnekenberg Teixeira (RA00336189)

---

## Research Question

Can a movie's commercial success be predicted **before release**, using only information available at that time?

A movie is labeled a **success** when its worldwide gross exceeds its production budget (`Worldwide_Gross > Production_Budget`), and a **flop** otherwise. This is a binary classification problem.

## Dataset

- **Source:** `vega_datasets.data.movies()` (3,201 movies, 16 columns).
- **Pre-release features used:** `Production_Budget`, release month (as season), `MPAA_Rating`, `Major_Genre`, `Source`, `Creative_Type`, `Distributor`.
- **Label inputs:** `Worldwide_Gross` and `Production_Budget` (gross is used only to build the label, never as a feature).

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn missingno vega_datasets
```

Then open and run `movie-success-prediction.ipynb` (tested on Google Colab).

---

## Exploratory Data Analysis

### Missingness
Missing values are **structured, not random**:
- `Source`, `Creative_Type` and `Distributor` tend to be missing together.
- When metadata is missing, `MPAA_Rating` is missing in 56–69% of cases (base rate 18.9%).
- When `MPAA_Rating` is missing, `Running_Time_min` is missing in 98.7% of cases.
- Movies without a rating are older (median year 1985 vs. 2003) and cheaper (median budget $7M vs. $24M).

### Data quality issues fixed
- **Century errors:** 22 classic films were dated 100 years in the future (e.g. *Gone with the Wind* as 2039). Years after 2011 were shifted back by 100.
- **Zero grosses:** 47 movies with `Worldwide_Gross = 0` were treated as missing.

### Key findings
- **Era effect:** films before 1996 succeed far more often (78–94%), reflecting survivorship bias and sparse coverage.
- **Runtime is a documentation artifact:** films with a known runtime succeed 79.5% of the time vs. 45.5% when missing, and the gap is unstable across years. A new film always has a known runtime, so this signal would not transfer.
- **"Unknown" metadata behaves the same way:** films with missing genre, source, creative type or distributor mostly flopped (20–26% success).
- **Seasonality:** summer (May–Jul) ≈ 75% success, holidays (Nov–Dec) ≈ 70%, September weakest (≈ 55%).

---

## Data Preparation

The final modeling frame (`model_df`) is rebuilt from the raw data in a single reproducible cell:

1. Fix century errors in `Release_Date`.
2. Drop rows with zero/missing gross or budget and build the `success` label.
3. Keep films from **1996 onward**.
4. Merge rare categories (`Open` → `Not Rated`; rare `Source`, `Creative_Type`, `Major_Genre` levels → `Other`).
5. Keep **complete cases** for categorical features (no `"Unknown"` levels, no missing-value indicators).
6. Exclude `Running_Time_min` from the features (kept only for a sensitivity check).

**Result:** 2,164 films, 66.7% successes.

---

## Modeling

### Time-based split
Train on the past, test on the future (cutoff = 80% quantile of release date):

| Split | Period | Films | Success rate |
|---|---|---|---|
| Train | 1996 – Apr 2007 | 1,731 | 65.7% |
| Test | Apr 2007 – 2010 | 433 | 70.4% |

Release year is used only to define the split and validation folds, never as a feature.

### Validation
**Expanding-window validation** inside the training set: for each year 2002–2007, train on all earlier years and validate on that year. Encoders are fit inside each fold to avoid leakage. Rare categorical levels are grouped by `OneHotEncoder(min_frequency=...)`.

### Choices
- **Month encoding:** numeric, one-hot and 3-level season were statistically tied; **season** was chosen for simplicity and domain grounding.
- **Hyperparameters:** grid over depth {2, 3, 4, 6} × min leaf {20, 40, 80}. Using the one-standard-error rule, **depth 4, min leaf 40** was selected (within noise of the best AUC, lowest variance).

---

## Results (Test Set)

| Model | Accuracy | Balanced acc. | ROC-AUC |
|---|---|---|---|
| Majority dummy | 0.704 | 0.500 | 0.500 |
| Budget-only tree (depth 4, leaf 40) | – | – | 0.533 |
| **Full tree (depth 4, leaf 40)** | 0.704 | 0.500 (0.566 at threshold 0.657) | **0.632** |

- **Paired bootstrap (2,000 resamples):** AUC gain over budget-only = **0.099**, 95% CI **[0.037, 0.160]**. The extra features add real ranking information.
- **Main features used by the tree:** production budget, small/independent distributors, summer release, Kids Fiction, Contemporary Fiction.
- **At the default 0.5 threshold** the tree predicts "success" for every film. Lowering the threshold to the training base rate (0.657) raises flop recall to 72%, but accuracy falls to 50% and flop precision is only 34%.

---

## Conclusion

Pre-release factors carry a **modest but real signal**: the tree ranks films by risk clearly better than budget alone, using season, distributor size and creative type. However, the signal is too weak to classify individual films reliably. This is expected, since success depends heavily on factors outside the data (cast, marketing, competition, reviews).

Strict leakage control (no runtime, no missingness indicators, time-based split, in-fold encoding) makes this modest result trustworthy rather than inflated.

## Limitations

- Single train/test cut with only 433 test films (AUC uncertainty ≈ ±0.05); cross-validation suggested a lower AUC (≈ 0.57).
- Data ends in 2010; budgets and grosses are in nominal dollars.
- Results apply only to films with complete metadata.
- Feature importances from a single tree are unstable and not causal.
