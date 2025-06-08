# Optuna LSTM Tuning Summary
- Date: 2025-06-08
- Subject: 7012
- Trials: 10
- Objective: Maximize F1 Score

---

## Best Trial
- **F1 Score**: 0.6361
- **Threshold**: 0.58
- **Accuracy**: 0.5925
- **Params**
  - `hidden_size`: 128
  - `num_layers`: 2
  - `dropout`: 0.46113270413265095
  - `bidirectional`: False
  - `learning_rate`: 0.00011982944130266843
  - `stride`: 1
  - `seq_len`: 64
  - `epochs`: 9
  - `batch_size`: 64
  - `weight_decay`: 1.3467657668402778e-05

---

## Top 5 Trials
| Trial | F1 Score | Threshold | Accuracy |
|-------|----------|-----------|----------|
| 3 | 0.6361 | 0.58 | 0.5925 |
| 9 | 0.6204 | 0.45 | 0.5826 |
| 6 | 0.6015 | 0.51 | 0.5566 |
| 8 | 0.5976 | 0.43 | 0.5536 |
| 2 | 0.5971 | 0.47 | 0.5516 |

---

## Visualizations
### Hyperparameter Importance
![F1 Importance](f1_importance_barplot.png)

### Correlation Heatmap
![Correlation with F1](corr_heatmap.png)

---

## Notes