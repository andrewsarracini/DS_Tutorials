from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV # Bayseian Optimization
from tpot import TPOTClassifier # Automated ML, Classification Tasks

# Hyperparameter Tuning:
def grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    ''' 
    Performs exhaustive grid search hyperparam tuning
    '''
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1) 
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def random_search_tuning(model, param_dist, X_train, y_train, cv=5, n_iter=50, scoring='accuracy'):
    ''' 
    Performs random search hyperparam tuning
    '''
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def bayesian_opt_tuning(model, search_spaces, X_train, y_train, cv=5, n_iter=50, scoring='accuracy'):
    ''' 
    Performs Bayesian optimization param tuning
    '''
    bayes_search = BayesSearchCV(model, search_spaces=search_spaces, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    bayes_search.fit(X_train, y_train)
    return bayes_search.get_params(deep=True)

#------------------------------------------------
# This will take hours/days to run, USE WITH CAUTION!!! 
def tpot_opt_tuning(X_train, y_train, X_test, y_test, generations=5, population_size=20, cv=5):
    '''
    Uses TPOT to find the best params
    Does not work like the other methods, generates a PYTHON FILE when complete
    *** CAUTION! TPOT can take hours/days to run!
    '''
    tpot = TPOTClassifier(generations=generations, population_size=population_size, cv=cv, verbosity=2, n_jobs=-1)
    tpot.fit(X_train, y_train)

    # BE AWARE: this generates a brand new python file that will have optimal params!
    return tpot.fitted_pipeline_, tpot.export('tpot_pipeline.py') 
