from src.tune import grand_tuner
from src.train import train_model
from src.helper import detect_class_imbalance, stratified_sample, detect_class_imbalance
from src.eval import eval_classification

def tune_and_train_full(model_class, model_name, X_train, y_train,
                        sample_frac=0.1, model_params=None,
                        X_test=None, y_test=None, dev_mode=False,
                        **tuner_kwargs): 
    
    """
    Full workflow:
    1. Sample data
    2. Tune on sample
    3. Save best params
    4. Train final model on full data
    """
    
    dev_mode = tuner_kwargs.pop("dev_mode", dev_mode)

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

    # DEV_MODE ONLY
    X_train_final = X_sample if dev_mode else X_train
    y_train_final = y_sample if dev_mode else y_train

    trained_models = train_model(X_train_final, y_train_final, {
        model_name: (model_class, merged_params)
    })

    trained_model = trained_models[model_name]

    if X_test is not None and y_test is not None:
        print(f'\n Running evaluation on test set...')

        is_imbalanced, minority_ratio = detect_class_imbalance(y_test)
        threshold = 0.25 if is_imbalanced else 0.5

        if is_imbalanced:
            print(f"⚠️ Imbalanced test set detected (Minority class = {minority_ratio:.2%})")
        else:
            print(f"Balanced test set detected, using default threshold: {threshold}")

        eval_classification(trained_model, X_test, y_test, threshold)

    else: 
        print("⚠️ No test set provided, skipping evaluation.\n")

    

    return trained_model, best_params