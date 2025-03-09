import os
import re


def citations(methods=None, filepath=None, format_type="bibtex"):
    """
    Generate references associated with the methods.

    Parameters
    ----------
    methods : str or list, optional
        Specifies the PreliZ methods or classes for which to display references.
        Use `"all"` to show references for all available methods/classes.
        Defaults to `None`, which displays PreliZ references.

    filepath : str or a file-like object
        Specifies the location to save the file.
        If ``None``, the result is returned as a string.

    format_type : str
       Specifies in which type the references will be displayed.
       Currently, only "bibtex" is supported.
       Defaults to ``bibtex``.

    """
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.bib")
    with open(ref_path, encoding="utf-8") as fr:
        content = fr.read()
    if methods == "all":
        keys = set(re.findall(r"@.*?\{(.*?),", content))
    elif methods is None:
        keys = {"PreliZ", "Icazatti2023"}
    else:
        keys = set()
        for method in methods:
            matches = set(re.findall(r":(?:cite|footcite):[tp]:`(.*?)`", method.__doc__))
            keys.update(matches)
    if format_type == "bibtex":
        cite_text = _citation_bibtex(content, keys)
        if filepath:
            with open(filepath, "w") as fw:
                fw.write(cite_text)
        else:
            return cite_text
    else:
        raise ValueError("Invalid value for format_type. Use 'bibtex'.")


def _citation_bibtex(content, keys):
    """Extract and return references in BibTeX format."""
    extracted_refs = []
    for key in keys:
        match = re.search(rf"(@\w+\{{\s*{key}\s*,.*?\n\}})", content, re.DOTALL)
        extracted_refs.append(match.group(1))
    return "\n".join(extracted_refs)
