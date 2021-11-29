# Differentiable Intonation Tools

The Differentiable Intonation Tools (`dit`) are a collection of Python functions to analyze the intonation in multitrack audio signals.

In particular, two measures are made available here, a tonal cost (the proximity to a tonal grid) and a harmonic cost (the perceptual dissonance between salient frequencies). The cost measures can be used to adapt intonation of multitrack audio signals in relation to each other, by minimizing the the cost using gradient descent (as exemplified in the notebook `examples.ipynb`).


## Installation

There is no pip package (yet?), so please clone this repository, navigate to the root folder and install with
```
python setup.py [install|develop]
```
or
```
pip install -e .
```

Required software packages: numpy (>= 1.17.0), scipy (>= 1.7.0), [librosa](https://github.com/librosa/librosa) (>= 0.8.0), [libtsm](https://github.com/meinardmueller/libtsm) (>= 1.1.0)

## Usage

Please see the notebook `examples.ipynb` for some usage examples. API documentation can be found in the code files.

More background and application examples can be found in the reference below and on the [accompanying website](https://www.audiolabs-erlangen.de/resources/MIR/2021-ISMIR-IntonationCostMeasure).


## Reference

If you use the Differentiable Intonation Tools in your research, please cite the following reference.

Simon Schwär, Sebastian Rosenzweig, and Meinard Müller. [A Differentiable Cost Measure for Intonation Processing in Polyphonic Music](https://archives.ismir.net/ismir2021/paper/000078.pdf). In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR): 626–633, 2021.


## License (MIT)

Copyright © 2021 Simon Schwär, Sebastian Rosenzweig, and Meinard Müller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgements

At time of writing, the authors of this software have been associated with the International Audio Laboratories Erlangen, a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS. The project was supported by the German Research Foundation (DFG MU 2686/12-1, MU 2686/13-1).

