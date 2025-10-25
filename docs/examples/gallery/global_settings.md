---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Global Settings

```{jupyter-execute}

import preliz as pz
```

Some of the PreliZ default values are regulated by `preliz.rcParams`, a class similar to a dictionary storing key-value pairs inspired by the one in matplotlib and ArviZ. It is similar to a dictionary and not a dictionary though because all keys are fixed, and each key has associated a validation function to help prevent setting nonsensical defaults.

## Preliz Configuration File

The `rcParams` class is generated and populated at import time. PreliZ checks several locations for a file named `prelizrc` and, if found, prefers those settings over the library ones.

The locations checked are the following:

1. Current working directory, {func}`os.getcwd`
2. Location indicated by {envvar}`PRELIZ_DATA` environment variable
3. The third and last location checked is OS dependent:
   - On Linux: `$XDG_CONFIG_HOME/preliz` if exists, otherwise `~/.config/preliz/`
   - Elsewhere: `~/.preliz/`

The file is a simple text file with a structure similar to the following:

```text
stats.ci_kind : eti
stats.ci_prob : 0.89
```

All available keys are listed below. The `prelizrc` file can have any subset of the keys, it isn't necessary to include them all. For those keys without a user defined default, the library one is used.

To find out the current settings, you can use:

```{jupyter-execute}

pz.rcParams
```

## Context manager 

A context manager is also available to temporarily change the default settings.

```{jupyter-execute}

with pz.rc_context({"plots.show_plot": False}):
    pz.maxent(pz.Normal())
```

The context manager accepts a dictionary, a file path, or both.
