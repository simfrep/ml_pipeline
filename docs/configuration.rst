===
Why
===

The underlying approach is to minimize actual coding when fitting and evaluating models.
The (variable) inputs are defined in a configuration yaml file that consists of

* Input/Output files and paths
* Target Variables
* "Bad" variables that should not be used in fitting, like IDs etc.
* Timeframes for Training/Validation/Test splits
* Modelling Pipeline

The configuration yaml file can be separated in the 3 sections

Input Section
=============

The first set of parameters describes the data used.

.. code-block:: yaml

    data: msft.parq
    dpath: 'data/'
    dbins: 'data_woe_bins.dat'
    ppath: 'plot_bins/'
    dvarsel: 'data_varsel.parq'
    target: 'target'
    bads: 
    ['Ticker', 'SimFinId',
        'Tickerinc', 'SimFinIdinc',
        'Currency', 'FiscalYear', 'FiscalPeriod', 'ReportDate', 'RestatedDate',
        'PublishDate', 'AdjClose_s1', 'spread',
        'target', 
     ]
    price_open: 'AdjClose'
    price_close: 'AdjClose_s1'


Pipeline Section
================

This section defines key steps the fitting procedure needs to run through, like

* Timeframes for data splits
* Rerunning previously run sections

.. code-block:: yaml

    begin_training: 2009-08-04
    end_training: 2017-12-31
    begin_valid: 2018-01-01
    end_valid: 2018-12-31
    begin_test: 2019-01-01
    end_test: 2020-09-30
    offset_lst: [0,48]
    offset_res: 'months'
    rerun_varsel: False
    rerun_binning: False
    refit_models: True
    use_binning: False # Allowed: True, False, Both




Preprocessing and Model section
===============================


.. code-block:: yaml

    # Optionally Run this function in a pipeline before model is fit
    # If use_binning is True no preprocessing is done
    preprocessing: 
    sklearn.preprocessing.RobustScaler:
    sklearn.preprocessing.StandardScaler:
    optbinning.BinningProcess:

    # Last model to create prediction
    models:
    sklearn.tree.DecisionTreeClassifier:
        criterion: ['gini']
        max_depth: [2,6,6,8,10]
    sklearn.ensemble.RandomForestClassifier:
        n_estimators: [500]  # [250,500,750,1000] 
        max_depth: [2,6,6,8,10]
