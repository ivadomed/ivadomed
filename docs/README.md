# Ivadomed Docs

## Installation

First, you will need to install the following requirements:

```
cd ivadomed
pip install -r requirements.txt
pip install sphinx_rtd_theme
```

## Build

To create the html pages from the `.rst` files:

```
cd ivadomed/docs
make html
```

Check out the `Makefile` for more information.

## View

Under `docs/build/html`, open `index.html` in your browser to preview
the `Sphinx` documentation.
