# Optuna LSTM Tuning Summary
- Date: 2025-06-08
- Subject: 7242
- Trials: 10
- Objective: Maximize F1 Score

---

## Best Trial
- **F1 Score**: 0.5366
- **Threshold**: 0.5
- **Accuracy**: 0.5298
- **Params**
  - `hidden_size`: 128
  - `num_layers`: 3
  - `dropout`: 0.32605054158312186
  - `bidirectional`: True
  - `learning_rate`: 0.0007451376549375596
  - `stride`: 1
  - `seq_len`: 64
  - `epochs`: 9
  - `batch_size`: 64
  - `weight_decay`: 0.0022291764931384015

---

## Top 5 Trials
| Trial | F1 Score | Threshold | Accuracy |
|-------|----------|-----------|----------|
| 8 | 0.5366 | 0.5 | 0.5298 |
| 5 | 0.5298 | 0.5 | 0.5146 |
| 3 | 0.4882 | 0.5 | 0.4930 |
| 6 | 0.4723 | 0.5 | 0.4537 |
| 7 | 0.4615 | 0.5 | 0.4623 |

---

## Visualizations
### Hyperparameter Importance
![F1 Importance](f1_importance_barplot.png)

### Correlation Heatmap
![Correlation with F1](corr_heatmap.png)

---

## Notes