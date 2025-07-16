.. Esper Morphogen documentation master file, created by
   sphinx-quickstart on Wed Jul 16 03:41:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Esper Morphogen Documentation
=============================

Esper Morphogen is a high-performance morphogenetic training platform that enables real-time AI model 
adaptation and optimization. It provides a comprehensive framework for dynamic model transformation 
with microsecond-latency execution and GPU-resident caching.

Features
--------

* **High-performance execution**: Microsecond-latency kernel execution with GPU-resident caching
* **Strategic intelligence**: Graph Neural Network-based policy system for morphogenetic control
* **Comprehensive services**: Database management, training services, and worker coordination
* **Type-safe configuration**: Pydantic-based configuration management
* **Scalable architecture**: Modular design supporting distributed deployments

Quick Start
-----------

To get started with Esper Morphogen, use the ``wrap()`` function to transform your PyTorch models:

.. code-block:: python

    from esper import wrap
    
    # Transform a standard PyTorch model
    morphogenic_model = wrap(your_pytorch_model)
    
    # The model now has morphogenetic capabilities
    output = morphogenic_model(input_tensor)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

