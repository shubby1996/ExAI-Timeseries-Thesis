# Model Comparison - Quick Summary

**Date**: December 11, 2025  
**Models Evaluated**: TFT vs NHiTS  
**Full Report**: [model_comparison.md](file:///c:/Uni%20Stuff/Semester%205/Thesis_SI/ShubhamThesis/data/docs/model_comparison.md)

---

## 🏆 Winner: TFT (Temporal Fusion Transformer)

TFT outperforms NHiTS across **all evaluation metrics**.

---

## 📊 Key Metrics Comparison

| Metric | TFT | NHiTS | Winner | Improvement |
|--------|-----|-------|--------|-------------|
| **MAE** (MW) | 0.211 | 0.228 | ✅ TFT | **7.5%** |
| **RMSE** (MW) | 0.259 | 0.298 | ✅ TFT | **13.1%** |
| **MAPE** (%) | 6.24% | 6.87% | ✅ TFT | **9.2%** |
| **R²** | 0.455 | 0.277 | ✅ TFT | **64.3%** |
| **PICP** (%) | 49.7% | 0.0% | ✅ TFT | ✅ Functional |
| **MIW** (MW) | 0.391 | 0.0 | ✅ TFT | ✅ Functional |
| **Quantile Loss** | 0.074 | 0.114 | ✅ TFT | **35.1%** |

---

## 💡 Key Insights

### Accuracy
- Both models achieve **excellent MAPE < 7%** (industry standard is < 10%)
- TFT's average error is only **211 kW** vs NHiTS's **228 kW**
- TFT explains **45.5%** of variance vs NHiTS's **27.7%**

### Uncertainty Quantification
- **TFT**: Provides meaningful confidence intervals (49.7% coverage)
- **NHiTS**: Failed to produce uncertainty estimates (0% coverage)
- This makes TFT critical for **risk-aware operational planning**

### Model Size
- **TFT**: 22.7 MB (smaller)
- **NHiTS**: 60.9 MB (2.7x larger)

---

## 🎯 Recommendations

### For Production
1. **Deploy TFT as primary model** - Superior accuracy + uncertainty quantification
2. **Use confidence intervals** - Enable risk-aware planning and decision-making
3. **Monitor PICP** - Consider recalibration if coverage drifts from target

### For Research
1. **Calibrate TFT uncertainty** - Current 49.7% PICP is below target 80%
2. **Debug NHiTS quantiles** - Investigate why uncertainty quantification failed
3. **Hyperparameter tuning** - Potential for further improvements

---

## 📈 Visual Evidence

### TFT Performance
- Strong tracking of actual consumption patterns
- Visible confidence intervals capturing variability
- Errors normally distributed with no systematic bias

### NHiTS Performance
- Reasonable tracking but more deviation
- No visible confidence intervals
- Wider error distribution with longer tails

**See full report for detailed visualizations**: [model_comparison.md](file:///c:/Uni%20Stuff/Semester%205/Thesis_SI/ShubhamThesis/data/docs/model_comparison.md)

---

## ✅ Conclusion

**TFT is the clear winner** for Nordbyen heat forecasting:
- ✅ Best accuracy (6.24% MAPE)
- ✅ Functional uncertainty quantification
- ✅ Smaller model size
- ✅ Better variance explanation (R² = 0.455)

Both models meet industry standards, but **TFT's uncertainty quantification capability makes it essential for operational planning** where understanding prediction confidence is critical.

---

**Next Steps**: See [task.md](file:///c:/Uni%20Stuff/Semester%205/Thesis_SI/ShubhamThesis/data/docs/task.md) for upcoming work including TCN debugging and thesis writing.
