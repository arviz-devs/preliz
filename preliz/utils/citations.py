import os
import re

try:
    from IPython.display import Markdown, display
except ImportError:
    pass


def citations(methods=None, show_as="bibtex"):
    """
    Generate references associated with the methods.

    Parameters
    ----------
    methods : A list of preliz methods
        Displays references for the specified methods or classes.
        Defaults to ``None``, which shows PreliZ's references.

    show_as : str
       Specifies in which type the references will be displayed.
       Currently, only "bibtex" is supported.
       Defaults to ``bibtex``.

    """
    if methods is None:
        keys = {"PreliZ", "Icazatti2023"}
    else:
        keys = set()
        for method in methods:
            matches = set(re.findall(r":(?:cite|footcite):[tp]:`(.*?)`", method.__doc__))
            keys.update(matches)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.bib")
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    if show_as == "bibtex":
        return _citation_bibtex(content, keys)
    else:
        raise ValueError("Invalid value for show_as. Use 'bibtex'.")


def _citation_bibtex(content, keys):
    """Display the references in a bibtex format."""
    gen_citations = []
    for key in keys:
        match = re.search(rf"(@\w+\{{\s*{key}\s*,.*?\n\}})", content, re.DOTALL)
        gen_citations.append(match.group(1))
    gen_citations = "\n".join(gen_citations)
    display(Markdown(f"```bibtex\n{gen_citations}\n```"))
    return gen_citations
