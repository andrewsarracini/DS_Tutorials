# Optuna LSTM Tuning Summary
- Date: 2025-05-23
- Subject: 7242
- Trials: 10
- Objective: Maximize F1 Score

---

## Best Trial
- **F1 Score**: 0.8210
- **Threshold**: 0.57
- **Accuracy**: 0.822
- **Params**
  - `hidden_size`: 64
  - `num_layers`: 2
  - `dropout`: 0.108466076468862
  - `bidirectional`: True
  - `learning_rate`: 0.00082609199251563
  - `stride`: 4
  - `seq_len`: 128
  - `epochs`: 13
  - `batch_size`: 32
  - `weight_decay`: 0.00010376177657048565

---

## Top 5 Trials
| Trial | F1 Score | Threshold | Accuracy |
|-------|----------|-----------|----------|
| 9 | 0.8210 | 0.57 | 0.8220 |
| 3 | 0.7956 | 0.63 | 0.7878 |
| 4 | 0.7614 | 0.58 | 0.7528 |
| 7 | 0.5798 | 0.56 | 0.5380 |
| 2 | 0.5733 | 0.56 | 0.5288 |

---

## Visualizations
### Hyperparameter Importance
![F1 Importance](f1_importance_barplot.png)

### Correlation Heatmap
![Correlation with F1](corr_heatmap.png)

---

## Notes