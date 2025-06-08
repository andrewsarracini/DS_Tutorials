# Optuna LSTM Tuning Summary
- Date: 2025-06-08
- Subject: 7012
- Trials: 10
- Objective: Maximize F1 Score

---

## Best Trial
- **F1 Score**: 0.3170
- **Threshold**: 0.5
- **Accuracy**: 0.2977
- **Params**
  - `hidden_size`: 64
  - `num_layers`: 2
  - `dropout`: 0.36965344602653816
  - `bidirectional`: True
  - `learning_rate`: 0.001720021822320989
  - `stride`: 2
  - `seq_len`: 128
  - `epochs`: 8
  - `batch_size`: 64
  - `weight_decay`: 0.003062273519768889

---

## Top 5 Trials
| Trial | F1 Score | Threshold | Accuracy |
|-------|----------|-----------|----------|
| 1 | 0.3170 | 0.5 | 0.2977 |
| 3 | 0.2487 | 0.5 | 0.2970 |
| 4 | 0.2156 | 0.5 | 0.2108 |
| 2 | 0.2099 | 0.5 | 0.2082 |
| 9 | 0.1779 | 0.5 | 0.2261 |

---

## Visualizations
### Hyperparameter Importance
![F1 Importance](f1_importance_barplot.png)

### Correlation Heatmap
![Correlation with F1](corr_heatmap.png)

---

## Notes