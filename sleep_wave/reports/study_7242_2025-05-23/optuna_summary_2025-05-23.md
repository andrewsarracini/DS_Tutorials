# Optuna LSTM Tuning Summary
- Date: 2025-05-23
- Subject: 7242
- Trials: 25
- Objective: Maximize F1 Score

---

## Best Trial
- **F1 Score**: 0.8273
- **Threshold**: 0.59
- **Accuracy**: 0.8302
- **Params**
  - `hidden_size`: 64
  - `num_layers`: 1
  - `dropout`: 0.13732882308497943
  - `bidirectional`: True
  - `learning_rate`: 0.0006893736626641891
  - `stride`: 4
  - `seq_len`: 128
  - `epochs`: 13
  - `batch_size`: 32
  - `weight_decay`: 9.911301531041695e-05

---

## Top 5 Trials
| Trial | F1 Score | Threshold | Accuracy |
|-------|----------|-----------|----------|
| 21 | 0.8273 | 0.59 | 0.8302 |
| 18 | 0.8266 | 0.63 | 0.8258 |
| 22 | 0.8242 | 0.6 | 0.8247 |
| 15 | 0.8117 | 0.58 | 0.8094 |
| 9 | 0.8035 | 0.67 | 0.8047 |

---

## Visualizations
### Hyperparameter Importance
![F1 Importance](f1_importance_barplot.png)

### Correlation Heatmap
![Correlation with F1](corr_heatmap.png)

---

## Notes