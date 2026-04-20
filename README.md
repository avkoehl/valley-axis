# valley-axis

Valley width is a key parameter to measure along rivers. Typically it is
measured using cross sections perpendicular to the valley centerline. A
weakness of this approach is that the valley centerline is not always well
defined, nor is it obvious how to associate river reaches to the valley cross
sections.

This package provides functions to:
- compute a valley centerline using a skeletonization approach tailored to the river network structure
- allocate regions on the valley floor to the valley centerline segments
- compute valley width within the valley centerline segments


![Valley width example](img/example.png)

## Installation

```bash
pip install git+https://github.com/avkoehl/valley-axis.git
```

With development dependencies:

```bash
git clone https://github.com/avkoehl/valley-axis.git
cd valley-axis
uv sync --extra dev
```

install jupyter kernel:
```bash
uv run python -m ipykernel install --user --name=valley-axis
```

## Usage

See example notebook: [example.ipynb](https://github.com/avkoehl/valley-axis/blob/main/examples/valley_axis_demo.ipynb)
