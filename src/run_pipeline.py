from src.tune import grand_tuner, save_best_params, stratified_sample, param_spaces
from src.train import train_model, load_best_params

def tune_and_train_full(model_class, model_name, X_full, y_full, sample_frac=0.1, **tuner_kwargs):
    """
    Full workflow:
    1. Sample data
    2. Tune on sample
    3. Save best params
    4. Train final model on full data
    """
    print(f"\n🔍 Sampling {sample_frac*100:.1f}% of data for tuning...")
    X_sample, y_sample = stratified_sample(X_full, y_full, sample_frac=sample_frac)

    # Instantiate model
    base_model = model_class()

    # Tune on sample
    best_model, best_params, _ = grand_tuner(
        model=base_model,
        X=X_sample,
        y=y_sample,
        param_grid=None,
        **tuner_kwargs
    )

    # Load and train on full data
    loaded_params = load_best_params(model_name)
    trained_models = train_model(X_full, y_full, {
        model_name: (model_class, loaded_params)
    })

    return trained_models[model_name]