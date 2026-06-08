import os

from gtts import gTTS

# List of terms to generate audio files for
names = {
    "en": [
        "AsymmetricLaplace",
        "Censored",
        "ExGaussian",
        "Exponential",
        "HalfCauchy",
        "HalfNormal",
        "HalfStudentT",
        "Hurdle",
        "InverseGamma",
        "Logistic",
        "LogLogistic",
        "LogitNormal",
        "LogNormal",
        "Mixture",
        "Normal",
        "Rice",
        "SkewNormal",
        "SkewStudentT",
        "StudentT",
        "Triangular",
        "Truncated",
        "TruncatedNormal",
        "Uniform",
        "Binomial",
        "Categorical",
        "DiscreteUniform",
        "Geometric",
        "HyperGeometric",
        "NegativeBinomial",
        "ZeroInflatedBinomial",
        "ZeroInflatedNegativeBinomial",
        "MultivariateNormal",
    ],
    "de": ["Bernoulli", "DiscreteWeibull", "Gumbel", "VonMises", "Wald", "Weibull"],
    "el": [
        ("Beta", "βήτα"),
        ("Betascaled", "βήταscaled"),
        ("BetaBinomial", "βήταsBinomial"),
        ("Gamma", "γάμμα"),
        ("chisquared", "χι-squared"),
    ],
    "fr": ["Cauchy", "Dirichlet", "Laplace", "Poisson", "ZeroInflatedPoisson"],
    "it": ["pareto"],
    "hi": ["Kumaraswamy"],
    "es": ["Moyal"],
}


output_dir = "audios"
os.makedirs(output_dir, exist_ok=True)

for lang, terms in names.items():
    for term in terms:
        if lang == "el":
            file, name = term
        else:
            file = name = term

        tts = gTTS(text=name, lang=lang)
        tts.save(f"{output_dir}/{file.lower()}.mp3")
