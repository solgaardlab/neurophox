# neurophox

The `neurophox` module is an open source machine learning and photonic simulation framework based on unitary mesh networks presented in [arxiv/1808.00458](https://arxiv.org/pdf/1808.00458.pdf). This library is meant to be a concise yet thorough treatment of mesh networks for broader use by the photonics and machine learning communities. All examples to replicate paper results are contained in separate repositories (see e.g. []).

The `neurophox` module currently provides unitary mesh layers in Tensorflow 2 (`neurophox.tensorflow.layers`) and we are working to make such layers available in PyTorch (`neurophox.torch.layers`) as well in a future release, and we will finalize these layers once PyTorch provides complex number support (contributions are welcome!). These layers can be used in any machine learning model and are equipped with Numpy equivalents (`neurophox.numpy.layers`) for testing and further visualization (e.g. light propagation visualizations for photonics simulations).

Scattering matrix models used in unitary mesh networks for photonics simulations are provided in `neurophox.components`. The models for all layers are fully defined in `neurophox.meshmodel`, which provides a general framework for efficient implementation of any unitary mesh network.

Keep in mind that the module is also under development and is not yet stable. We welcome pull requests and contributions from the broader community.

Please see our [wiki](https://github.com/solgaardlab/neurophox/wiki) for more details about our library. We have information for how to run code examples and/or contribute to our codebase.