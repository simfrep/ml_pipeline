<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">ML_Pipeline (WIP)</h3>

  <p align="center">
    A library making fitting models hassle free
    <br />
    <a href="https://github.com/simfrep/ml_pipeline"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/simfrep/ml_pipeline">View Demo</a>
    ·
    <a href="https://github.com/simfrep/ml_pipeline/issues">Report Bug</a>
    ·
    <a href="https://github.com/simfrep/ml_pipeline/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The majority of ML applications is doing some standard data transformations and fitting sklearn models.

In my experience I often want to try wildly different preprocessing/modelling steps and see what works
and what not.

I wrote this to make my life easier and to get rid of the annoying parts and to avoid any beginner trappings (e.g. feature transformation on whole datset etc.)

This project minimizes writing python codes and substitues it by flexible yaml configuration files

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Apart from standard python libraries this package uses
* `scikit-learn` (duh...) using pipelines, metrics and models
* `dill` for storing binary model files
* `munch` for accessing yaml-config entries 

If you use conda you can use this yaml file

    templates/environment.yaml

to install a python environment to run any of the templates or examples listed

```sh
conda env create -f environment.yaml
```

### Installation

TODO python package installation


<!-- USAGE EXAMPLES -->
## Usage

Just run this code from a notebook located in one of the example folders
```python
import yaml
import munch
import logging

logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s",level=logging.INFO)

from mlpipeline import MLPipeline
# Load Config
c = munch.munchify(yaml.load(open('msft.yaml','r')))
mp = MLPipeline(c)
# Run defined models
mp.mltrain_loop()
```

One advantage of this package is that it immediately writes each model with the defined metrics into the filesystem.
So you can easily inspect or work with the fitted models while the training loop is still running.

Create a separate notebook and run this code to see the best models
```python
import yaml
import munch
import logging

logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s",level=logging.INFO)

from mlpipeline import MLPipeline
# Load Config
c = munch.munchify(yaml.load(open('msft.yaml','r')))
mp = MLPipeline(c)
# Run defined models
best = mp.extract_best_model()
for key in best.keys():
    asc = False
    if key == 'sklearn.metrics.log_loss':
        asc = True
    display(best[key].sort_values(by='valid',ascending=asc).head(5))
```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the BSD-3 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Simon Preis - simfrep@gmail.com

Project Link: [https://github.com/simfrep/ml_pipeline](https://github.com/simfrep/ml_pipeline)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template/)
* [Img Shields](https://shields.io)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/simfrep/ml_pipeline.svg?style=for-the-badge
[contributors-url]: https://github.com/simfrep/ml_pipeline/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/simfrep/ml_pipeline.svg?style=for-the-badge
[forks-url]: https://github.com/simfrep/ml_pipeline/network/members
[stars-shield]: https://img.shields.io/github/stars/simfrep/ml_pipeline.svg?style=for-the-badge
[stars-url]: https://github.com/simfrep/ml_pipeline/stargazers
[issues-shield]: https://img.shields.io/github/issues/simfrep/ml_pipeline.svg?style=for-the-badge
[issues-url]: https://github.com/simfrep/ml_pipeline/issues
[license-shield]: https://img.shields.io/github/license/simfrep/ml_pipeline.svg?style=for-the-badge
[license-url]: https://github.com/simfrep/ml_pipeline/blob/master/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/simon-preis-489518114
[product-screenshot]: images/screenshot.png
