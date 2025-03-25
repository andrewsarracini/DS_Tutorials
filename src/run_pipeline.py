from src.tune import grand_tuner
from src.train import train_model
from src.helper import save_best_params, load_best_params, serialize_params, stratified_sample, param_spaces, dynamic_param_grid
from src.eval import eval_classification

def tune_and_train_full(model_class, model_name, X_train, y_train,
                        sample_frac=0.1, model_params=None,
                        X_test=None, y_test=None, **tuner_kwargs): 
    
    """
    Full workflow:
    1. Sample data
    2. Tune on sample
    3. Save best params
    4. Train final model on full data
    """

    X_sample, y_sample = stratified_sample(X_train, y_train, sample_frac=sample_frac)

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

    trained_models = train_model(X_train, y_train, {
        model_name: (model_class, merged_params)
    })

    trained_model = trained_models[model_name]

    if X_test is not None and y_test is not None:
        print(f'\n Running evaluation on test set...\n')
        eval_classification(trained_model, X_test, y_test)
    else: 
        print("⚠️ No test set provided, skipping evaluation.\n")

    return trained_model, best_params