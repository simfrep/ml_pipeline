=====
About
=====

As can be seen from the Example *Run ML Pipeline* the calling of functions is generic
and the only variable component is the *configuration.yaml* that is used when 
initializing the class

Run ML Pipeline
===============

.. code-block:: python

    import yaml
    from fit import MLPipeline

    # Load configuration yaml
    c = yaml.load(open('msft.yaml','r'),Loader=yaml.FullLoader)

    # Initiate class
    mp = MLPipeline(c)

    # Run Variable Selection (using scorecardpy)
    mp.iv_varsel()

    # Create WoE binning (using scorecardpy)
    mp.woe_bins()

    # Loop through all combinations of Models and Training timeframes
    mp.mltrain_loop()