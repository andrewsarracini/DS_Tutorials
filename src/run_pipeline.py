from src.tune import grand_tuner
from src.train import train_model
from src.helper import save_best_params, load_best_params, serialize_params, stratified_sample, param_spaces, dynamic_param_grid


def tune_and_train_full(model_class, model_name, X_full, y_full, sample_frac=0.1, model_params=None, **tuner_kwargs):
    """
    Full workflow:
    1. Sample data
    2. Tune on sample
    3. Save best params
    4. Train final model on full data
    """

    X_sample, y_sample = stratified_sample(X_full, y_full, sample_frac=sample_frac)

    # Instantiate model
    base_model = model_class(**(model_params or {}))

    # INITIATE THE GRAND TUNER
    best_model, best_params, _ = grand_tuner(
        model=base_model,
        X=X_sample,
        y=y_sample,
        param_grid=None,
        **tuner_kwargs
    )

    # Load and train on full data
    # Merge custom added model_params + best_params
    merged_params = (model_params or {}).copy()
    merged_params.update(best_params)

    trained_models = train_model(X_full, y_full, {
        model_name: (model_class, merged_params)
    })

    return trained_models[model_name], best_params