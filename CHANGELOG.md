<a id="0.14.0"></a>
# [0.14.0](https://github.com/arviz-devs/preliz/releases/tag/0.14.0) - 2025-01-05

## What's Changed

###  New Features
* Add rcParams by [@aloctavodia](https://github.com/aloctavodia) in [#615](https://github.com/arviz-devs/preliz/pull/615)

### Maintenance and bug fixes
* Add test for modes by [@aloctavodia](https://github.com/aloctavodia) in [#613](https://github.com/arviz-devs/preliz/pull/613)
* Do not expose posterior_to_prior by [@aloctavodia](https://github.com/aloctavodia) in [#614](https://github.com/arviz-devs/preliz/pull/614)


### Documentation
* Distributions Gallery: Add pronunciation to ZIB, ZINB and ZIP by [@aleicazatti](https://github.com/aleicazatti) in [#606](https://github.com/arviz-devs/preliz/pull/606)
* Distributions Gallery: Add Dirichlet by [@aleicazatti](https://github.com/aleicazatti) in [#608](https://github.com/arviz-devs/preliz/pull/608)
* Distributions Gallery: Add MvNormal by [@aleicazatti](https://github.com/aleicazatti) in [#609](https://github.com/arviz-devs/preliz/pull/609)
* Distributions Gallery: Add Mixture by [@aleicazatti](https://github.com/aleicazatti) in [#610](https://github.com/arviz-devs/preliz/pull/610)
* Distributions Gallery: Add Hurdle by [@aleicazatti](https://github.com/aleicazatti) in [#611](https://github.com/arviz-devs/preliz/pull/611)
* Fix Roulette docstring by [@aloctavodia](https://github.com/aloctavodia) in [#607](https://github.com/arviz-devs/preliz/pull/607)
* Fix hurdle docstring by [@aloctavodia](https://github.com/aloctavodia) in [#612](https://github.com/arviz-devs/preliz/pull/612)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.13.0...0.14.0

[Changes][0.14.0]


<a id="0.13.0"></a>
# [0.13.0](https://github.com/arviz-devs/preliz/releases/tag/0.13.0) - 2024-12-13

## What's Changed

###  New Features
* Add mode argument to maxent by [@aloctavodia](https://github.com/aloctavodia) in [#603](https://github.com/arviz-devs/preliz/pull/603)

### Maintenance and bug fixes
* refactor ppe and related functions by [@aloctavodia](https://github.com/aloctavodia) in [#597](https://github.com/arviz-devs/preliz/pull/597)

### Documentation
* Distribution Gallery: Add Geometric by [@aleicazatti](https://github.com/aleicazatti) in [#595](https://github.com/arviz-devs/preliz/pull/595)
* Distributions Gallery: Improve layout Distribution Pages by [@aleicazatti](https://github.com/aleicazatti) in [#596](https://github.com/arviz-devs/preliz/pull/596)
* Distributions Gallery: Apply new layout to distribution pages by [@aleicazatti](https://github.com/aleicazatti) in [#598](https://github.com/arviz-devs/preliz/pull/598)
* Distribution Gallery: Add HyperGeometric by [@aleicazatti](https://github.com/aleicazatti) in [#600](https://github.com/arviz-devs/preliz/pull/600)
* Distributions Gallery: Add ZIB by [@aleicazatti](https://github.com/aleicazatti) in [#601](https://github.com/arviz-devs/preliz/pull/601)
* Distributions Gallery: Add ZINB by [@aleicazatti](https://github.com/aleicazatti) in [#602](https://github.com/arviz-devs/preliz/pull/602)
* Distributions Gallery: Add ZIP by [@aleicazatti](https://github.com/aleicazatti) in [#605](https://github.com/arviz-devs/preliz/pull/605)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.12.0...0.13.0

[Changes][0.13.0]


<a id="0.12.0"></a>
# [0.12.0](https://github.com/arviz-devs/preliz/releases/tag/0.12.0) - 2024-11-21

## What's Changed

###  New Features
* Make `quartile_int` a class by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#570](https://github.com/arviz-devs/preliz/pull/570)
* Add `Mixture` modifier by [@aloctavodia](https://github.com/aloctavodia) in [#572](https://github.com/arviz-devs/preliz/pull/572)
* Add `to_bambi` method by [@aloctavodia](https://github.com/aloctavodia) in [#578](https://github.com/arviz-devs/preliz/pull/578)

### Maintenance and bug fixes
* Refactor `ppe` by [@aloctavodia](https://github.com/aloctavodia) in [#579](https://github.com/arviz-devs/preliz/pull/579)
* fix corner case `Weibull` pdf by [@aloctavodia](https://github.com/aloctavodia) in [#583](https://github.com/arviz-devs/preliz/pull/583)
* fix corner case hdi `Censored` by [@aloctavodia](https://github.com/aloctavodia) in [#590](https://github.com/arviz-devs/preliz/pull/590)
* Circular intervals for `VonMises` by [@aloctavodia](https://github.com/aloctavodia) in [#592](https://github.com/arviz-devs/preliz/pull/592)
* `Dirichlet`: more robust joint pdf for larger values of alpha by [@aloctavodia](https://github.com/aloctavodia) in [#593](https://github.com/arviz-devs/preliz/pull/593)

### Documentation
* Distributions Gallery: Add Triangular by [@aleicazatti](https://github.com/aleicazatti) in [#571](https://github.com/arviz-devs/preliz/pull/571)
* Distributions Gallery: Add TruncatedNormal by [@aleicazatti](https://github.com/aleicazatti) in [#573](https://github.com/arviz-devs/preliz/pull/573)
* Distributions Gallery: Add VonMises by [@aleicazatti](https://github.com/aleicazatti) in [#574](https://github.com/arviz-devs/preliz/pull/574)
* Distributions Gallery: Add Wald by [@aleicazatti](https://github.com/aleicazatti) in [#581](https://github.com/arviz-devs/preliz/pull/581)
* Distributions Gallery: Add Weibull by [@aleicazatti](https://github.com/aleicazatti) in [#582](https://github.com/arviz-devs/preliz/pull/582)
* Distributions Gallery: Add BetaBinomial by [@aleicazatti](https://github.com/aleicazatti) in [#584](https://github.com/arviz-devs/preliz/pull/584)
* Distribution Gallery: Add Binomial by [@aleicazatti](https://github.com/aleicazatti) in [#586](https://github.com/arviz-devs/preliz/pull/586)
* Distribution Gallery: Add DiscreteUniform by [@aleicazatti](https://github.com/aleicazatti) in [#589](https://github.com/arviz-devs/preliz/pull/589)
* Distribution Gallery: Add Categorical by [@aleicazatti](https://github.com/aleicazatti) in [#588](https://github.com/arviz-devs/preliz/pull/588)
* Distribution Gallery: Add DiscreteWeibull by [@aleicazatti](https://github.com/aleicazatti) in [#591](https://github.com/arviz-devs/preliz/pull/591)
* Reorganize and update docs by [@aloctavodia](https://github.com/aloctavodia) in [#594](https://github.com/arviz-devs/preliz/pull/594)

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.11.0...0.12.0

[Changes][0.12.0]


<a id="0.11.0"></a>
# [0.11.0](https://github.com/arviz-devs/preliz/releases/tag/0.11.0) - 2024-10-25

## What's Changed

## New Features
* Add combine_roulette function by [@aloctavodia](https://github.com/aloctavodia) in [#555](https://github.com/arviz-devs/preliz/pull/555)
* Add combine function by [@aloctavodia](https://github.com/aloctavodia) in [#557](https://github.com/arviz-devs/preliz/pull/557)
* mle: handle multidimensional samples by [@aloctavodia](https://github.com/aloctavodia) in [#568](https://github.com/arviz-devs/preliz/pull/568)

## Documentation
* Add ChangeLog to website by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#562](https://github.com/arviz-devs/preliz/pull/562)
* Distribution Gallery: Add Laplace by [@aleicazatti](https://github.com/aleicazatti) in [#553](https://github.com/arviz-devs/preliz/pull/553)
* Distributions Gallery: Add LogitNormal by [@aleicazatti](https://github.com/aleicazatti) in [#560](https://github.com/arviz-devs/preliz/pull/560)
* Distributions Gallery: Add Pareto by [@aleicazatti](https://github.com/aleicazatti) in [#563](https://github.com/arviz-devs/preliz/pull/563)
* Distributions Gallery: Add Rice by [@aleicazatti](https://github.com/aleicazatti) in [#565](https://github.com/arviz-devs/preliz/pull/565)
* Distributions Gallery: Add SkewNormal by [@aleicazatti](https://github.com/aleicazatti) in [#566](https://github.com/arviz-devs/preliz/pull/566)
* Distributions Gallery: Add Moyal by [@aleicazatti](https://github.com/aleicazatti) in [#561](https://github.com/arviz-devs/preliz/pull/561)
* Distributions Gallery: Add SkewStudentT by [@aleicazatti](https://github.com/aleicazatti) in [#567](https://github.com/arviz-devs/preliz/pull/567)
* Distribution Gallery: Add Log-Logistic by [@aleicazatti](https://github.com/aleicazatti) in [#556](https://github.com/arviz-devs/preliz/pull/556)
* Update tagline  by [@aloctavodia](https://github.com/aloctavodia) in [#558](https://github.com/arviz-devs/preliz/pull/558)



**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.10.0...0.11.0

[Changes][0.11.0]


<a id="0.10.0"></a>
# [0.10.0](https://github.com/arviz-devs/preliz/releases/tag/0.10.0) - 2024-10-01

## What's Changed

### New Features
* Extend "to_pymc" to transformed variables by [@aloctavodia](https://github.com/aloctavodia) in [#544](https://github.com/arviz-devs/preliz/pull/544)
* Roulette is a class now and fitted distribution can be accessed with the attribute `.dist`  by [@aloctavodia](https://github.com/aloctavodia) in [#546](https://github.com/arviz-devs/preliz/pull/546)

### Maintenance and bug fixes

* Fix bounds and psi param for Hurdle by [@aloctavodia](https://github.com/aloctavodia) in [#537](https://github.com/arviz-devs/preliz/pull/537)
* Remove ArviZ dependency by [@aloctavodia](https://github.com/aloctavodia) in [#536](https://github.com/arviz-devs/preliz/pull/536)
* PPA: several fixes by [@aloctavodia](https://github.com/aloctavodia) in [#549](https://github.com/arviz-devs/preliz/pull/549)

### Documentation
* Add reference to PriorDB by [@aloctavodia](https://github.com/aloctavodia) in [#535](https://github.com/arviz-devs/preliz/pull/535)
* Distribution Gallery: Add HalfCauchy by [@aleicazatti](https://github.com/aleicazatti) in [#538](https://github.com/arviz-devs/preliz/pull/538)
* Distribution Gallery: Add HalfNormal by [@aleicazatti](https://github.com/aleicazatti) in [#540](https://github.com/arviz-devs/preliz/pull/540)
* Distribution Gallery: Add HalfStudentT by [@aleicazatti](https://github.com/aleicazatti) in [#541](https://github.com/arviz-devs/preliz/pull/541)
* Distributions Gallery: Add InverseGamma by [@aleicazatti](https://github.com/aleicazatti) in [#542](https://github.com/arviz-devs/preliz/pull/542)
* Distribution Gallery: Add Kumaraswamy by [@aleicazatti](https://github.com/aleicazatti) in [#548](https://github.com/arviz-devs/preliz/pull/548)
* Add bounds to gallery images by [@aloctavodia](https://github.com/aloctavodia) in [#539](https://github.com/arviz-devs/preliz/pull/539)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.9.1...0.10.0

[Changes][0.10.0]


<a id="0.9.1"></a>
# [0.9.1](https://github.com/arviz-devs/preliz/releases/tag/0.9.1) - 2024-09-10

## What's Changed
* clarify where optimization results are stored by [@aloctavodia](https://github.com/aloctavodia) in [#525](https://github.com/arviz-devs/preliz/pull/525)
* Add docs for conda installation by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#528](https://github.com/arviz-devs/preliz/pull/528)
* Add posterior_to_prior example by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#527](https://github.com/arviz-devs/preliz/pull/527)
* Distribution Gallery: Beta Alternative Parametrization by [@aleicazatti](https://github.com/aleicazatti) in [#529](https://github.com/arviz-devs/preliz/pull/529)
* Add NegativeBinomial and Bernoulli distributions to Gallery by [@aloctavodia](https://github.com/aloctavodia) in [#530](https://github.com/arviz-devs/preliz/pull/530)
* Add pronunciation audios to gallery by [@aloctavodia](https://github.com/aloctavodia) in [#531](https://github.com/arviz-devs/preliz/pull/531)
* Distribution Gallery: ALD alternative parametrization by [@aleicazatti](https://github.com/aleicazatti) in [#532](https://github.com/arviz-devs/preliz/pull/532)
* Distribution Gallery: Alternative parametrization by [@aleicazatti](https://github.com/aleicazatti) in [#533](https://github.com/arviz-devs/preliz/pull/533)
* Raise error if distribution passed to maxent or quartile is frozen by [@aloctavodia](https://github.com/aloctavodia) in [#534](https://github.com/arviz-devs/preliz/pull/534)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.9.0...0.9.1

[Changes][0.9.1]


<a id="0.9.0"></a>
# [0.9.0](https://github.com/arviz-devs/preliz/releases/tag/0.9.0) - 2024-08-16

## What's Changed

### New Features
* Add posterior_to_prior function by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#508](https://github.com/arviz-devs/preliz/pull/508)
* posterior_to_prior: accept idata return string with PyMC model by [@aloctavodia](https://github.com/aloctavodia) in [#512](https://github.com/arviz-devs/preliz/pull/512)
* Add params_dict attribute by [@aloctavodia](https://github.com/aloctavodia) in [#521](https://github.com/arviz-devs/preliz/pull/521)
* Add to_pymc method to distributions by [@aloctavodia](https://github.com/aloctavodia) in [#523](https://github.com/arviz-devs/preliz/pull/523)
* Add bambi support for posterior_to_prior by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#522](https://github.com/arviz-devs/preliz/pull/522)

### Maintenance and bug fixes
* Solve for pymc>=5.16.0 by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#485](https://github.com/arviz-devs/preliz/pull/485)
* Add legend argument to multivariate distributions by [@aloctavodia](https://github.com/aloctavodia) in [#488](https://github.com/arviz-devs/preliz/pull/488)
* Don't run tests if only docs folder changes by [@OriolAbril](https://github.com/OriolAbril) in [#495](https://github.com/arviz-devs/preliz/pull/495)
* Clean namespace and add the missing elements to API doc by [@aloctavodia](https://github.com/aloctavodia) in [#511](https://github.com/arviz-devs/preliz/pull/511)
* Update tests for posterior_to_prior by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#515](https://github.com/arviz-devs/preliz/pull/515)

### Documentation
* Add Distribution Gallery  by [@aleicazatti](https://github.com/aleicazatti) in [#465](https://github.com/arviz-devs/preliz/pull/465)
* Add note about PyMC and Bambi installation by [@aloctavodia](https://github.com/aloctavodia) in [#472](https://github.com/arviz-devs/preliz/pull/472)
* Update normal distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#477](https://github.com/arviz-devs/preliz/pull/477)
* Add Beta distribution to the gallery by [@aloctavodia](https://github.com/aloctavodia) in [#478](https://github.com/arviz-devs/preliz/pull/478)
* Add Student's T distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#480](https://github.com/arviz-devs/preliz/pull/480)
* Gallery: Add script to generate cover images by [@aloctavodia](https://github.com/aloctavodia) in [#479](https://github.com/arviz-devs/preliz/pull/479)
* Add logistic distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#481](https://github.com/arviz-devs/preliz/pull/481)
* Add Cauchy distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#483](https://github.com/arviz-devs/preliz/pull/483)
* Fix Logistic docstring by [@aleicazatti](https://github.com/aleicazatti) in [#484](https://github.com/arviz-devs/preliz/pull/484)
* Add LogNormal distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#486](https://github.com/arviz-devs/preliz/pull/486)
* Add Exponential distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#487](https://github.com/arviz-devs/preliz/pull/487)
* Add Multivariate images to the distribution gallery by [@aloctavodia](https://github.com/aloctavodia) in [#489](https://github.com/arviz-devs/preliz/pull/489)
* Gallery: Add Censored and  Truncated distributions. by [@aloctavodia](https://github.com/aloctavodia) in [#490](https://github.com/arviz-devs/preliz/pull/490)
* Add Poisson to the distribution gallery by [@aloctavodia](https://github.com/aloctavodia) in [#491](https://github.com/arviz-devs/preliz/pull/491)
* Gallery: Add uniform distribution page by [@aleicazatti](https://github.com/aleicazatti) in [#494](https://github.com/arviz-devs/preliz/pull/494)
* Gallery: Add asymmetric Laplace distribution by [@aloctavodia](https://github.com/aloctavodia) in [#497](https://github.com/arviz-devs/preliz/pull/497)
* Gallery: Add beta scaled by [@aloctavodia](https://github.com/aloctavodia) in [#499](https://github.com/arviz-devs/preliz/pull/499)
* Gallery: Uniform, remove repeated paragraph by [@aleicazatti](https://github.com/aleicazatti) in [#501](https://github.com/arviz-devs/preliz/pull/501)
* Gallery: Default to API docs for missing cards by [@aloctavodia](https://github.com/aloctavodia) in [#502](https://github.com/arviz-devs/preliz/pull/502)
* Distribution Gallery: add ChiSquared by [@aleicazatti](https://github.com/aleicazatti) in [#504](https://github.com/arviz-devs/preliz/pull/504)
* Add tabs for predictive_explorer in observed_space_examples_all.rst by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#505](https://github.com/arviz-devs/preliz/pull/505)
* Distribution Gallery: Add Gamma by [@aleicazatti](https://github.com/aleicazatti) in [#506](https://github.com/arviz-devs/preliz/pull/506)
* use PreliZ style by [@aloctavodia](https://github.com/aloctavodia) in [#509](https://github.com/arviz-devs/preliz/pull/509)
* Distributions Gallery: Add ExGaussian by [@aleicazatti](https://github.com/aleicazatti) in [#507](https://github.com/arviz-devs/preliz/pull/507)
* Distributions Gallery: Add Gumbel by [@aleicazatti](https://github.com/aleicazatti) in [#514](https://github.com/arviz-devs/preliz/pull/514)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.8.1...0.9.0

[Changes][0.9.0]


<a id="0.8.1"></a>
# [0.8.1](https://github.com/arviz-devs/preliz/releases/tag/0.8.1) - 2024-06-17

## What's Changed

* Fix Missing distributions in api_reference.rst by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#467](https://github.com/arviz-devs/preliz/pull/467)
* Improve Docs for predictive functions by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#468](https://github.com/arviz-devs/preliz/pull/468)
* Remove sphinx_thebe conf and dependency by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#469](https://github.com/arviz-devs/preliz/pull/469)
* fix moments labels by [@aloctavodia](https://github.com/aloctavodia) in [#471](https://github.com/arviz-devs/preliz/pull/471)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.8.0...0.8.1

[Changes][0.8.1]


<a id="0.8.0"></a>
# [0.8.0](https://github.com/arviz-devs/preliz/releases/tag/0.8.0) - 2024-06-11

## What's Changed

### New Features

* Add pymc support to predictive explorer by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#450](https://github.com/arviz-devs/preliz/pull/450)
* Add support for engine="auto" and bambi models in predictive explorer by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#455](https://github.com/arviz-devs/preliz/pull/455)
* Add example for bambi and pymc to observed_space_examples by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#456](https://github.com/arviz-devs/preliz/pull/456)
* use g format string for distribution repr by [@aloctavodia](https://github.com/aloctavodia) in [#459](https://github.com/arviz-devs/preliz/pull/459)
* improve hdi computation and interval representation by [@aloctavodia](https://github.com/aloctavodia) in [#458](https://github.com/arviz-devs/preliz/pull/458)

### Maintenance and bug fixes

* Fix TruncatedNormal Docs by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#446](https://github.com/arviz-devs/preliz/pull/446)
* improve interval report by [@aloctavodia](https://github.com/aloctavodia) in [#457](https://github.com/arviz-devs/preliz/pull/457)
* fix small issues in docstrings by [@aloctavodia](https://github.com/aloctavodia) in [#454](https://github.com/arviz-devs/preliz/pull/454)


Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.7.0...0.8.0

[Changes][0.8.0]


<a id="0.7.0"></a>
# [0.7.0](https://github.com/arviz-devs/preliz/releases/tag/0.7.0) - 2024-05-20

## What's Changed


### New Features
* Add LogLogistic distribution by [@aloctavodia](https://github.com/aloctavodia) in [#438](https://github.com/arviz-devs/preliz/pull/438)
* Adds support for multiple targets with weights to ppe by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#442](https://github.com/arviz-devs/preliz/pull/442)

### Maintenance and bug fixes 
* relax mle test by [@aloctavodia](https://github.com/aloctavodia) in [#441](https://github.com/arviz-devs/preliz/pull/441)
* Add tests to ppe by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#436](https://github.com/arviz-devs/preliz/pull/436)
* improve x_val by [@aloctavodia](https://github.com/aloctavodia) in [#440](https://github.com/arviz-devs/preliz/pull/440)



**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.6.3...0.7.0

[Changes][0.7.0]


<a id="0.6.3"></a>
# [0.6.3](https://github.com/arviz-devs/preliz/releases/tag/0.6.3) - 2024-05-15

## What's Changed
* Censored and Truncated. Allow bounds to be vectors by [@aloctavodia](https://github.com/aloctavodia) in [#431](https://github.com/arviz-devs/preliz/pull/431)
* Add SkewStudentT by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#432](https://github.com/arviz-devs/preliz/pull/432)
* Fix mle bugs in some distributions by [@aloctavodia](https://github.com/aloctavodia) in [#434](https://github.com/arviz-devs/preliz/pull/434) and [#435](https://github.com/arviz-devs/preliz/pull/435)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.6.2...0.6.3

[Changes][0.6.3]


<a id="0.6.2"></a>
# [0.6.2](https://github.com/arviz-devs/preliz/releases/tag/0.6.2) - 2024-05-09

## What's Changed
* add metadata to package so it can be added to conda forge by [@OriolAbril](https://github.com/OriolAbril) in [#430](https://github.com/arviz-devs/preliz/pull/430)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.6.1...0.6.2

[Changes][0.6.2]


<a id="0.6.1"></a>
# [0.6.1](https://github.com/arviz-devs/preliz/releases/tag/0.6.1) - 2024-05-08

## What's Changed
* uneven x_vals by [@aloctavodia](https://github.com/aloctavodia) in [#424](https://github.com/arviz-devs/preliz/pull/424)
* fix binomial logpdf at bounds by [@aloctavodia](https://github.com/aloctavodia) in [#425](https://github.com/arviz-devs/preliz/pull/425)
* ignore modifier distributions when mapping pymc to preliz by [@aloctavodia](https://github.com/aloctavodia) in [#426](https://github.com/arviz-devs/preliz/pull/426)
* use xlogx/y/1py and fix logpdf values outside of the support by [@aloctavodia](https://github.com/aloctavodia) in [#428](https://github.com/arviz-devs/preliz/pull/428)



**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.6.0...0.6.1

[Changes][0.6.1]


<a id="0.6.0"></a>
# [0.6.0](https://github.com/arviz-devs/preliz/releases/tag/0.6.0) - 2024-04-26

## What's Changed

### New features

* Add Hurdle distribution by [@aloctavodia](https://github.com/aloctavodia) in [#398](https://github.com/arviz-devs/preliz/pull/398)

### Add custom distribution (not SciPy wrappers):
* Add Logistic by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#394](https://github.com/arviz-devs/preliz/pull/394)
* Add Gumbel by [@aloctavodia](https://github.com/aloctavodia) in [#395](https://github.com/arviz-devs/preliz/pull/395)
* Add Pareto by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#396](https://github.com/arviz-devs/preliz/pull/396)
* Add Cauchy by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#401](https://github.com/arviz-devs/preliz/pull/401)
* Add LogNormal by [@aloctavodia](https://github.com/aloctavodia) in [#402](https://github.com/arviz-devs/preliz/pull/402)
* Add Kumaraswamy by [@aloctavodia](https://github.com/aloctavodia) in [#403](https://github.com/arviz-devs/preliz/pull/403)
* Add ChiSquared distribution by [@aloctavodia](https://github.com/aloctavodia) in [#404](https://github.com/arviz-devs/preliz/pull/404)
* Add Moyal by [@aloctavodia](https://github.com/aloctavodia) in [#405](https://github.com/arviz-devs/preliz/pull/405)
* Add LogitNormal by [@aloctavodia](https://github.com/aloctavodia) in [#406](https://github.com/arviz-devs/preliz/pull/406)
* Add hypergeometric by [@aloctavodia](https://github.com/aloctavodia) in [#407](https://github.com/arviz-devs/preliz/pull/407)
* Add HalfCauchy by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#409](https://github.com/arviz-devs/preliz/pull/409)
* Add DiscreteWeibull by [@aloctavodia](https://github.com/aloctavodia) in [#410](https://github.com/arviz-devs/preliz/pull/410)
* Add BetaBinomial by [@aloctavodia](https://github.com/aloctavodia) in [#411](https://github.com/arviz-devs/preliz/pull/411)
* Add TruncatedNormal by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#414](https://github.com/arviz-devs/preliz/pull/414)
* Add SkewNormal by [@aloctavodia](https://github.com/aloctavodia) in [#415](https://github.com/arviz-devs/preliz/pull/415)
* Add Rice by [@aloctavodia](https://github.com/aloctavodia) in [#416](https://github.com/arviz-devs/preliz/pull/416)
* Add BetaScaled by [@aloctavodia](https://github.com/aloctavodia) in [#417](https://github.com/arviz-devs/preliz/pull/417)
* Add ExGaussian by [@aloctavodia](https://github.com/aloctavodia) in [#419](https://github.com/arviz-devs/preliz/pull/419)

### Bug fixes and maintenance
* Add env.yml files for conda/mamba install by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#400](https://github.com/arviz-devs/preliz/pull/400)
* Remove transition code by [@aloctavodia](https://github.com/aloctavodia) in [#420](https://github.com/arviz-devs/preliz/pull/420)
* improve logpdf ExGaussian by [@aloctavodia](https://github.com/aloctavodia) in [#421](https://github.com/arviz-devs/preliz/pull/421)
* Use neg_logpdf by [@aloctavodia](https://github.com/aloctavodia) in [#422](https://github.com/arviz-devs/preliz/pull/422)
* Fix doc style for ZIB by [@aloctavodia](https://github.com/aloctavodia) in [#412](https://github.com/arviz-devs/preliz/pull/412)
* Remove interpolation for plot_pdf of discrete variables by [@aloctavodia](https://github.com/aloctavodia) in https://github.com/arviz-devs/preliz/pull

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.5.0...0.6.0

[Changes][0.6.0]


<a id="0.5.0"></a>
# [0.5.0](https://github.com/arviz-devs/preliz/releases/tag/0.5.0) - 2024-04-12

## What's Changed

### New features
* Add Truncated and Censored distributions  by [@aloctavodia](https://github.com/aloctavodia) in [#370](https://github.com/arviz-devs/preliz/pull/370) and in [#372](https://github.com/arviz-devs/preliz/pull/372)

### Bug fixes and maintenance 
* Fix bug for predictive_explorer and hist by [@aloctavodia](https://github.com/aloctavodia) in [#381](https://github.com/arviz-devs/preliz/pull/381)
* Beta: rename kappa to nu by [@aloctavodia](https://github.com/aloctavodia) in [#385](https://github.com/arviz-devs/preliz/pull/385)
* Add Python 3.12 to tests by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#390](https://github.com/arviz-devs/preliz/pull/390)

### Add custom distribution (not SciPy wrappers):
* Add VonMises by [@aloctavodia](https://github.com/aloctavodia) in [#376](https://github.com/arviz-devs/preliz/pull/376)
* Add StudentT and HalfStudentT by [@aloctavodia](https://github.com/aloctavodia) in [#379](https://github.com/arviz-devs/preliz/pull/379)
* Add InverseGamma by [@aloctavodia](https://github.com/aloctavodia) in [#382](https://github.com/arviz-devs/preliz/pull/382)
* Add Wald by [@aloctavodia](https://github.com/aloctavodia) in [#384](https://github.com/arviz-devs/preliz/pull/384)
* Add Gamma by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#387](https://github.com/arviz-devs/preliz/pull/387)
* Add Uniform and DiscreteUniform by [@aloctavodia](https://github.com/aloctavodia) in [#388](https://github.com/arviz-devs/preliz/pull/388)
* Add Categorical by [@aloctavodia](https://github.com/aloctavodia) in [#389](https://github.com/arviz-devs/preliz/pull/389)
* Add Triangular by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#391](https://github.com/arviz-devs/preliz/pull/391)
* Add Geometric by [@aloctavodia](https://github.com/aloctavodia) in [#392](https://github.com/arviz-devs/preliz/pull/392)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.4.1...0.5.0

[Changes][0.5.0]


<a id="0.4.1"></a>
# [0.4.1](https://github.com/arviz-devs/preliz/releases/tag/0.4.1) - 2024-03-20

## What's Changed
* Ipython should be optional by [@aloctavodia](https://github.com/aloctavodia) in [#369](https://github.com/arviz-devs/preliz/pull/369)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.4.0...0.4.1

[Changes][0.4.1]


<a id="0.4.0"></a>
# [0.4.0](https://github.com/arviz-devs/preliz/releases/tag/0.4.0) - 2024-03-20

## What's Changed
* Add plot_interactive to MvNormal by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#337](https://github.com/arviz-devs/preliz/pull/337)
* Dirichlet Elicitation by [@nishant42491](https://github.com/nishant42491) in [#327](https://github.com/arviz-devs/preliz/pull/327)
* Add faster Normal implementation by [@aloctavodia](https://github.com/aloctavodia) in [#344](https://github.com/arviz-devs/preliz/pull/344)
* Add ppe method for predictive elicitation (very experimental) by [@aloctavodia](https://github.com/aloctavodia) in [#336](https://github.com/arviz-devs/preliz/pull/336)
* Add faster HalfNormal distribution by [@aloctavodia](https://github.com/aloctavodia) in [#346](https://github.com/arviz-devs/preliz/pull/346)
* Add faster Poisson by [@aloctavodia](https://github.com/aloctavodia) in [#347](https://github.com/arviz-devs/preliz/pull/347)
* Add faster Bernoulli and Binomial by [@aloctavodia](https://github.com/aloctavodia) in [#348](https://github.com/arviz-devs/preliz/pull/348)
* Add faster Beta by [@aloctavodia](https://github.com/aloctavodia) in [#350](https://github.com/arviz-devs/preliz/pull/350)
* Add faster NegativeBinomial by [@aloctavodia](https://github.com/aloctavodia) in [#351](https://github.com/arviz-devs/preliz/pull/351)
* Add faster Weibull by [@aloctavodia](https://github.com/aloctavodia) in [#353](https://github.com/arviz-devs/preliz/pull/353)
* Add neg_log_pdf private method by [@aloctavodia](https://github.com/aloctavodia) in [#354](https://github.com/arviz-devs/preliz/pull/354)
* Add faster Exponential by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#355](https://github.com/arviz-devs/preliz/pull/355)
* Add faster Laplace by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#356](https://github.com/arviz-devs/preliz/pull/356)
* Add n_points argument to x_vals. by [@aloctavodia](https://github.com/aloctavodia) in [#357](https://github.com/arviz-devs/preliz/pull/357)
* Add faster ZINB by [@aloctavodia](https://github.com/aloctavodia) in [#359](https://github.com/arviz-devs/preliz/pull/359)
* Add faster ZIP by [@aloctavodia](https://github.com/aloctavodia) in [#360](https://github.com/arviz-devs/preliz/pull/360)
* Add faster ZIB by [@aloctavodia](https://github.com/aloctavodia) in [#362](https://github.com/arviz-devs/preliz/pull/362)
* Add faster Asymmetric Laplace by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#364](https://github.com/arviz-devs/preliz/pull/364)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.8...0.4.0

[Changes][0.4.0]


<a id="0.3.8"></a>
# [0.3.8](https://github.com/arviz-devs/preliz/releases/tag/0.3.8) - 2024-02-27

## What's Changed
* Add example for plot_func by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#311](https://github.com/arviz-devs/preliz/pull/311)
* Added quartile_int to `Direct elicitation in 1D by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#312](https://github.com/arviz-devs/preliz/pull/312)
* Adding tests for ppa by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#313](https://github.com/arviz-devs/preliz/pull/313)
* Add elicitation of Beta distribution with bounds and mode by [@nishant42491](https://github.com/nishant42491) in [#309](https://github.com/arviz-devs/preliz/pull/309)
* Add checkboxes to control distributions in roulette by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#318](https://github.com/arviz-devs/preliz/pull/318)
* Make default mass values consistent, invert order fmt-mass, check fmt is a string by [@aloctavodia](https://github.com/aloctavodia) in [#314](https://github.com/arviz-devs/preliz/pull/314)
* Update roulette plot and gif by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#322](https://github.com/arviz-devs/preliz/pull/322)
* Add beta mode tests by [@nishant42491](https://github.com/nishant42491) in [#323](https://github.com/arviz-devs/preliz/pull/323)
* Updated ppa widgets to include references by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#328](https://github.com/arviz-devs/preliz/pull/328)
* Fix MLE for halfnormal by [@aloctavodia](https://github.com/aloctavodia) in [#329](https://github.com/arviz-devs/preliz/pull/329)
* Remove predictive_finder by [@aloctavodia](https://github.com/aloctavodia) in [#330](https://github.com/arviz-devs/preliz/pull/330)
* Add plot_interactive to Dirichlet by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#332](https://github.com/arviz-devs/preliz/pull/332)



## New Contributors
* [@rohanbabbar04](https://github.com/rohanbabbar04) made their first contribution in [#311](https://github.com/arviz-devs/preliz/pull/311)
* [@nishant42491](https://github.com/nishant42491) made their first contribution in [#309](https://github.com/arviz-devs/preliz/pull/309)

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.7...0.3.8

[Changes][0.3.8]


<a id="0.3.7"></a>
# [0.3.7](https://github.com/arviz-devs/preliz/releases/tag/0.3.7) - 2024-01-31

## What's Changed
* refactor ppa by [@aloctavodia](https://github.com/aloctavodia) in [#295](https://github.com/arviz-devs/preliz/pull/295)
* Refactor PPA to improve internal documentation by [@aloctavodia](https://github.com/aloctavodia) in [#296](https://github.com/arviz-devs/preliz/pull/296)
* Use systematic resampling for predictive finder by [@aloctavodia](https://github.com/aloctavodia) in [#297](https://github.com/arviz-devs/preliz/pull/297)
* ensure integer x-axis for pmf by [@aloctavodia](https://github.com/aloctavodia) in [#298](https://github.com/arviz-devs/preliz/pull/298)
* fix bug beta distribution sigma/kappa by [@aloctavodia](https://github.com/aloctavodia) in [#299](https://github.com/arviz-devs/preliz/pull/299)
* Bambi parser (very experimental)) by [@aloctavodia](https://github.com/aloctavodia) in [#300](https://github.com/arviz-devs/preliz/pull/300)
* handle comments by [@aloctavodia](https://github.com/aloctavodia) in [#301](https://github.com/arviz-devs/preliz/pull/301)
* add reference values to predictive explorer by [@aloctavodia](https://github.com/aloctavodia) in [#302](https://github.com/arviz-devs/preliz/pull/302)
* check psi valid values for ZeroInflated distributions by [@aloctavodia](https://github.com/aloctavodia) in [#304](https://github.com/arviz-devs/preliz/pull/304)
* add references by [@aloctavodia](https://github.com/aloctavodia) in [#305](https://github.com/arviz-devs/preliz/pull/305)
* Override entropy of TruncatedNormal  by [@aloctavodia](https://github.com/aloctavodia) in [#306](https://github.com/arviz-devs/preliz/pull/306)
* predictive_explorer: Add option for custom plot by [@aloctavodia](https://github.com/aloctavodia) in [#307](https://github.com/arviz-devs/preliz/pull/307)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.6...0.3.7

[Changes][0.3.7]


<a id="0.3.6"></a>
# [0.3.6](https://github.com/arviz-devs/preliz/releases/tag/0.3.6) - 2023-10-27

## What's Changed
* unpin matplotlib by [@aloctavodia](https://github.com/aloctavodia) in [#285](https://github.com/arviz-devs/preliz/pull/285)
* fix a bug in roulette and improve scale by [@aloctavodia](https://github.com/aloctavodia) in [#286](https://github.com/arviz-devs/preliz/pull/286)
* add more distributions to roulette by [@aloctavodia](https://github.com/aloctavodia) in [#287](https://github.com/arviz-devs/preliz/pull/287)
* add quartile_int function by [@aloctavodia](https://github.com/aloctavodia) in [#288](https://github.com/arviz-devs/preliz/pull/288)
* customize roulette by [@aloctavodia](https://github.com/aloctavodia) in [#291](https://github.com/arviz-devs/preliz/pull/291)
* rename predictive_sliders to predictive_explorer refactor and new features by [@aloctavodia](https://github.com/aloctavodia) in [#292](https://github.com/arviz-devs/preliz/pull/292)
* add predictive_finder method (experimental) by [@aloctavodia](https://github.com/aloctavodia) in [#293](https://github.com/arviz-devs/preliz/pull/293)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.5...0.3.6

[Changes][0.3.6]


<a id="0.3.5"></a>
# [0.3.5](https://github.com/arviz-devs/preliz/releases/tag/0.3.5) - 2023-10-03

## What's Changed
* silence warnings by [@aloctavodia](https://github.com/aloctavodia) in [#280](https://github.com/arviz-devs/preliz/pull/280)
* extend exclude list ppa  by [@aloctavodia](https://github.com/aloctavodia) in [#282](https://github.com/arviz-devs/preliz/pull/282)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.4...0.3.5

[Changes][0.3.5]


<a id="0.3.4"></a>
# [0.3.4](https://github.com/arviz-devs/preliz/releases/tag/0.3.4) - 2023-09-21

## What's Changed
* Fix LaTeX umlaut by [@olexandr-konovalov](https://github.com/olexandr-konovalov) in [#277](https://github.com/arviz-devs/preliz/pull/277)

## New Contributors
* [@olexandr-konovalov](https://github.com/olexandr-konovalov) made their first contribution in [#277](https://github.com/arviz-devs/preliz/pull/277)

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.3...0.3.4

[Changes][0.3.4]


<a id="0.3.3"></a>
# [0.3.3](https://github.com/arviz-devs/preliz/releases/tag/0.3.3) - 2023-09-21

## What's Changed
* fix for scipy 1.11.1 by [@aloctavodia](https://github.com/aloctavodia) in [#272](https://github.com/arviz-devs/preliz/pull/272)
* allow singular covariance matrix  by [@aloctavodia](https://github.com/aloctavodia) in [#274](https://github.com/arviz-devs/preliz/pull/274)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.2...0.3.3

[Changes][0.3.3]


<a id="0.3.2"></a>
# [0.3.2](https://github.com/arviz-devs/preliz/releases/tag/0.3.2) - 2023-08-02

## What's Changed
* conditionally import ipywidgets by [@aloctavodia](https://github.com/aloctavodia) in [#268](https://github.com/arviz-devs/preliz/pull/268)
* drop python 3.8 add 3.11 by [@aloctavodia](https://github.com/aloctavodia) in [#270](https://github.com/arviz-devs/preliz/pull/270)



**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.1...0.3.2

[Changes][0.3.2]


<a id="0.3.1"></a>
# [0.3.1](https://github.com/arviz-devs/preliz/releases/tag/0.3.1) - 2023-07-08

## What's Changed
* add color and alpha arguments to plot methods by [@aloctavodia](https://github.com/aloctavodia) in [#240](https://github.com/arviz-devs/preliz/pull/240)
* add DiscreteWeibull  by [@aleicazatti](https://github.com/aleicazatti) in [#239](https://github.com/arviz-devs/preliz/pull/239)
* add bounds to optimize_moments by [@aloctavodia](https://github.com/aloctavodia) in [#243](https://github.com/arviz-devs/preliz/pull/243)
* add figsize argument to plot_interactive by [@aloctavodia](https://github.com/aloctavodia) in [#244](https://github.com/arviz-devs/preliz/pull/244)
* rename `fixed_lim` to `xy_lim` by [@aleicazatti](https://github.com/aleicazatti) in [#245](https://github.com/arviz-devs/preliz/pull/245)
* add tests for DiscreteWeibull by [@aleicazatti](https://github.com/aleicazatti) in [#242](https://github.com/arviz-devs/preliz/pull/242)
* fix entropy and moments of DiscreteWeibull by [@aloctavodia](https://github.com/aloctavodia) in [#249](https://github.com/arviz-devs/preliz/pull/249)
* use koay method for estimation of Rice parameters by [@aloctavodia](https://github.com/aloctavodia) in [#251](https://github.com/arviz-devs/preliz/pull/251)
* add opt example by [@aloctavodia](https://github.com/aloctavodia) in [#252](https://github.com/arviz-devs/preliz/pull/252)
* add overview to readme and landing page by [@aloctavodia](https://github.com/aloctavodia) in [#257](https://github.com/arviz-devs/preliz/pull/257)
* quartile init value and message by [@aloctavodia](https://github.com/aloctavodia) in [#256](https://github.com/arviz-devs/preliz/pull/256)
* specify model used by ppa by [@aloctavodia](https://github.com/aloctavodia) in [#258](https://github.com/arviz-devs/preliz/pull/258)
* add dependency bundles by [@aleicazatti](https://github.com/aleicazatti) in [#259](https://github.com/arviz-devs/preliz/pull/259)
* pin scipy version by [@aloctavodia](https://github.com/aloctavodia) in [#265](https://github.com/arviz-devs/preliz/pull/265)
* predictive slider: better automatic range by [@aloctavodia](https://github.com/aloctavodia) in [#264](https://github.com/arviz-devs/preliz/pull/264)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.3.0...0.3.1

[Changes][0.3.1]


<a id="0.3.0"></a>
# [0.3.0](https://github.com/arviz-devs/preliz/releases/tag/0.3.0) - 2023-04-19

## What's Changed
* Add optimize_moments internal function by [@aloctavodia](https://github.com/aloctavodia) in [#212](https://github.com/arviz-devs/preliz/pull/212)
* Add HyperGeometric distribution by [@aleicazatti](https://github.com/aleicazatti) in [#215](https://github.com/arviz-devs/preliz/pull/215)
* Fix bug maxent plot_kwargs by [@aloctavodia](https://github.com/aloctavodia) in [#216](https://github.com/arviz-devs/preliz/pull/216)
* Use absolute loss for optimize_moments by [@aloctavodia](https://github.com/aloctavodia) in [#217](https://github.com/arviz-devs/preliz/pull/217)
* Rename Student to StudentT by [@aloctavodia](https://github.com/aloctavodia) in [#219](https://github.com/arviz-devs/preliz/pull/219)
* Add ZeroInflatedBinomial by [@aleicazatti](https://github.com/aleicazatti) in [#220](https://github.com/arviz-devs/preliz/pull/220)
* Add ZeroInflatedNegativeBinomial by [@aleicazatti](https://github.com/aleicazatti) in [#222](https://github.com/arviz-devs/preliz/pull/222)
* Add initial support for MultiVariate Variables by [@aloctavodia](https://github.com/aloctavodia) in [#224](https://github.com/arviz-devs/preliz/pull/224)
* Add alternative parametrization for Wald by [@aleicazatti](https://github.com/aleicazatti) in [#230](https://github.com/arviz-devs/preliz/pull/230)
* Add alternative parametrization for MvNormal by [@aleicazatti](https://github.com/aleicazatti) in [#232](https://github.com/arviz-devs/preliz/pull/232)
* Add alternative parametrization for Exponential by [@aleicazatti](https://github.com/aleicazatti) in [#234](https://github.com/arviz-devs/preliz/pull/234)
* Add Kumaraswamy distribution by [@aleicazatti](https://github.com/aleicazatti) in [#231](https://github.com/arviz-devs/preliz/pull/231)
* Faster KDE predictive sliders  by [@aloctavodia](https://github.com/aloctavodia) in [#237](https://github.com/arviz-devs/preliz/pull/237)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.2.0...0.3.0

[Changes][0.3.0]


<a id="0.2.0"></a>
# [0.2.0](https://github.com/arviz-devs/preliz/releases/tag/0.2.0) - 2023-02-14

## What's Changed
* support non_scalar parameters for rvs, cdf and ppf methods by [@aloctavodia](https://github.com/aloctavodia) in [#162](https://github.com/arviz-devs/preliz/pull/162)
* Add test plot_interactive and roulette by [@aloctavodia](https://github.com/aloctavodia) in [#163](https://github.com/arviz-devs/preliz/pull/163) and in [#164](https://github.com/arviz-devs/preliz/pull/164)
* Refactor utils by [@aloctavodia](https://github.com/aloctavodia) in [#167](https://github.com/arviz-devs/preliz/pull/167)
* add predictive_sliders function by [@aloctavodia](https://github.com/aloctavodia) in [#168](https://github.com/arviz-devs/preliz/pull/168)
* DOCS: small changes to layout, name of notebook, fix typos, improve wording by [@aloctavodia](https://github.com/aloctavodia) in [#170](https://github.com/arviz-devs/preliz/pull/170) and in [@aloctavodia](https://github.com/aloctavodia) in [#171](https://github.com/arviz-devs/preliz/pull/171)
* Predictive sliders improve parser by [@aloctavodia](https://github.com/aloctavodia) in [#173](https://github.com/arviz-devs/preliz/pull/173)
* Make ppa PPL agnostic by [@aloctavodia](https://github.com/aloctavodia) in [#181](https://github.com/arviz-devs/preliz/pull/181)
* Add Bernoulli by [@aleicazatti](https://github.com/aleicazatti) in [#182](https://github.com/arviz-devs/preliz/pull/182)
* MLE: return index for sorting distributions by [@aloctavodia](https://github.com/aloctavodia) in [#184](https://github.com/arviz-devs/preliz/pull/184)
* add AICc to mle by [@aloctavodia](https://github.com/aloctavodia) in [#186](https://github.com/arviz-devs/preliz/pull/186)
* PPA: small fixes and refactor by [@aloctavodia](https://github.com/aloctavodia) in [#188](https://github.com/arviz-devs/preliz/pull/188)
* try fixing widgets rendering by [@OriolAbril](https://github.com/OriolAbril) in [#187](https://github.com/arviz-devs/preliz/pull/187)
* PPA: panel for the "total" prior predictive distribution, automatic selection, and family innovation by [@aloctavodia](https://github.com/aloctavodia) in [#190](https://github.com/arviz-devs/preliz/pull/190)
* make  ipympl a general requierement by [@aloctavodia](https://github.com/aloctavodia) in [#191](https://github.com/arviz-devs/preliz/pull/191)
* predictive_slider: correct support for histograms of discrete variables by [@aloctavodia](https://github.com/aloctavodia) in [#196](https://github.com/arviz-devs/preliz/pull/196)
* Add Geometric distribution by [@aleicazatti](https://github.com/aleicazatti) in [#197](https://github.com/arviz-devs/preliz/pull/197)
* Point interval: use HDI as default by [@aloctavodia](https://github.com/aloctavodia) in [#198](https://github.com/arviz-devs/preliz/pull/198)
* Add BetaBinomial distribution by [@aleicazatti](https://github.com/aleicazatti) in [#200](https://github.com/arviz-devs/preliz/pull/200)
* Add Rice distribution by [@aleicazatti](https://github.com/aleicazatti) in [#203](https://github.com/arviz-devs/preliz/pull/203)
* add categorical by [@aloctavodia](https://github.com/aloctavodia) in [#205](https://github.com/arviz-devs/preliz/pull/205)
* add logitnormal distribution by [@aloctavodia](https://github.com/aloctavodia) in [#206](https://github.com/arviz-devs/preliz/pull/206) and in [#210](https://github.com/arviz-devs/preliz/pull/210)
* Add ZeroInflatedPoisson by [@aloctavodia](https://github.com/aloctavodia) in [#209](https://github.com/arviz-devs/preliz/pull/209)


[Changes][0.2.0]


<a id="0.1.1"></a>
# [0.1.1](https://github.com/arviz-devs/preliz/releases/tag/0.1.1) - 2022-12-31

## What's Changed
* Refactor plot_interactive, enforce correct parameter types by [@aloctavodia](https://github.com/aloctavodia) in [#157](https://github.com/arviz-devs/preliz/pull/157)
* Small fixes [@aloctavodia](https://github.com/aloctavodia) in [#158](https://github.com/arviz-devs/preliz/pull/158) and  [#160](https://github.com/arviz-devs/preliz/pull/160) and [#159](https://github.com/arviz-devs/preliz/pull/159)

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.1.0...0.1.1

[Changes][0.1.1]


<a id="0.1.0"></a>
# [0.1.0](https://github.com/arviz-devs/preliz/releases/tag/0.1.0) - 2022-12-24

## What's Changed
* Add alternative parametrization for halfnormal distribution by [@aleicazatti](https://github.com/aleicazatti) in [#106](https://github.com/arviz-devs/preliz/pull/106)
* Fix warning message for ambiguous parametrization. Normal and Student by [@aleicazatti](https://github.com/aleicazatti) in [#108](https://github.com/arviz-devs/preliz/pull/108)
* Don't add labels if legend is disabled in plot_pdf/cdf/ppf by [@aloctavodia](https://github.com/aloctavodia) in [#109](https://github.com/arviz-devs/preliz/pull/109)
* Add alternative parametrization for SkewNormal distribution by [@aleicazatti](https://github.com/aleicazatti) in [#111](https://github.com/arviz-devs/preliz/pull/111)
* Add test for plots by [@aloctavodia](https://github.com/aloctavodia) in [#110](https://github.com/arviz-devs/preliz/pull/110)
* Add from_tau and to_tau functions to continuous module by [@aleicazatti](https://github.com/aleicazatti) in [#113](https://github.com/arviz-devs/preliz/pull/113)
* PPA: automatically create pymc_to_preliz dict by [@aloctavodia](https://github.com/aloctavodia) in [#114](https://github.com/arviz-devs/preliz/pull/114)
* Add alternative parametrization for HalfStudent distribution by [@aleicazatti](https://github.com/aleicazatti) in [#115](https://github.com/arviz-devs/preliz/pull/115)
* Rename from_tau and to_tau functions as the more general from_precision and to_precision. by [@aleicazatti](https://github.com/aleicazatti) in [#117](https://github.com/arviz-devs/preliz/pull/117)
* Add Logistic distribution by [@aleicazatti](https://github.com/aleicazatti) in [#118](https://github.com/arviz-devs/preliz/pull/118)
* Expose pdf, cdf and ppf by [@aloctavodia](https://github.com/aloctavodia) in [#119](https://github.com/arviz-devs/preliz/pull/119)
* Add Gumbel distribution by [@aleicazatti](https://github.com/aleicazatti) in [#120](https://github.com/arviz-devs/preliz/pull/120)
* Add Moyal distribution by [@aleicazatti](https://github.com/aleicazatti) in [#121](https://github.com/arviz-devs/preliz/pull/121)
* Add alternative parametrization to negative binomial by [@aloctavodia](https://github.com/aloctavodia) in [#122](https://github.com/arviz-devs/preliz/pull/122)
* Add gif of ppa example by [@aloctavodia](https://github.com/aloctavodia) in [#123](https://github.com/arviz-devs/preliz/pull/123)
* ppa: allow non-random initialization by [@aloctavodia](https://github.com/aloctavodia) in [#124](https://github.com/arviz-devs/preliz/pull/124)
* PPA: add representations; ecdf, histogram and option to set/unset sharex by [@aloctavodia](https://github.com/aloctavodia) in [#126](https://github.com/arviz-devs/preliz/pull/126)
* Add test quartile by [@aloctavodia](https://github.com/aloctavodia) in [#130](https://github.com/arviz-devs/preliz/pull/130)
* Specify support for some distributions in the docs by [@aleicazatti](https://github.com/aleicazatti) in [#133](https://github.com/arviz-devs/preliz/pull/133)
* Remove unnecessary calls to rv_frozen method by [@aloctavodia](https://github.com/aloctavodia) in [#134](https://github.com/arviz-devs/preliz/pull/134)
* Allow fixing arbitrary parameters for maxent and quartile by [@aloctavodia](https://github.com/aloctavodia) in [#136](https://github.com/arviz-devs/preliz/pull/136)
* Add ExGaussian distribution by [@aloctavodia](https://github.com/aloctavodia) in [#137](https://github.com/arviz-devs/preliz/pull/137)
* Add triangular distribution by [@aleicazatti](https://github.com/aleicazatti) in [#138](https://github.com/arviz-devs/preliz/pull/138)
* Add docstring examples to maxent and quartile, update notebook to show how to fix parameters by [@aloctavodia](https://github.com/aloctavodia) in [#139](https://github.com/arviz-devs/preliz/pull/139)
* Fix roulette by [@aloctavodia](https://github.com/aloctavodia) in [#142](https://github.com/arviz-devs/preliz/pull/142)
* ppa: add argument to specify reference values by [@aloctavodia](https://github.com/aloctavodia) in [#143](https://github.com/arviz-devs/preliz/pull/143)
* Add interactive method to distributions by [@aloctavodia](https://github.com/aloctavodia) in [#145](https://github.com/arviz-devs/preliz/pull/145)
* Rename interactive to plot_interactive, define integer or float slider as needed, add example by [@aloctavodia](https://github.com/aloctavodia) in [#146](https://github.com/arviz-devs/preliz/pull/146)
* Improve maximum likelihood estimation for NegativeBinomial by [@aloctavodia](https://github.com/aloctavodia) in [#147](https://github.com/arviz-devs/preliz/pull/147)
* Fix typo in README by [@aleicazatti](https://github.com/aleicazatti) in [#148](https://github.com/arviz-devs/preliz/pull/148)
* Increase tolerance Pareto test by [@aloctavodia](https://github.com/aloctavodia) in [#149](https://github.com/arviz-devs/preliz/pull/149)
* Allow fixing alt parameters by [@aloctavodia](https://github.com/aloctavodia) in [#150](https://github.com/arviz-devs/preliz/pull/150)
* Add logo by [@aloctavodia](https://github.com/aloctavodia) in [#151](https://github.com/arviz-devs/preliz/pull/151)
* Remove params_report attribute by [@aloctavodia](https://github.com/aloctavodia) in [#153](https://github.com/arviz-devs/preliz/pull/153)
* Add AsymmetricLaplace by [@aleicazatti](https://github.com/aleicazatti) in [#154](https://github.com/arviz-devs/preliz/pull/154)

**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.0.3...0.1.0

[Changes][0.1.0]


<a id="0.0.3"></a>
# [0.0.3](https://github.com/arviz-devs/preliz/releases/tag/0.0.3) - 2022-11-22

## What's Changed
* Add alternative parametrization for beta distribution by [@aloctavodia](https://github.com/aloctavodia) in [#94](https://github.com/arviz-devs/preliz/pull/94)
* Update readme by [@aleicazatti](https://github.com/aleicazatti) in [#95](https://github.com/arviz-devs/preliz/pull/95)
* Add alternative parametrization for normal distribution by [@aleicazatti](https://github.com/aleicazatti) in [#97](https://github.com/arviz-devs/preliz/pull/97)
* update installation instructions by [@aloctavodia](https://github.com/aloctavodia) in [#96](https://github.com/arviz-devs/preliz/pull/96)
* Fix bug by [@aleicazatti](https://github.com/aleicazatti) in [#98](https://github.com/arviz-devs/preliz/pull/98)
* Add alternative parametrization for Student distribution by [@aleicazatti](https://github.com/aleicazatti) in [#99](https://github.com/arviz-devs/preliz/pull/99)
* Add alternative parametrization for gamma distribution by [@aleicazatti](https://github.com/aleicazatti) in [#100](https://github.com/arviz-devs/preliz/pull/100)
* Add ChiSquared distribution by [@aleicazatti](https://github.com/aleicazatti) in [#101](https://github.com/arviz-devs/preliz/pull/101)
* Add alternative parametrization for inverse gamma distribution by [@aleicazatti](https://github.com/aleicazatti) in [#103](https://github.com/arviz-devs/preliz/pull/103)
* Add VonMises distribution by [@aleicazatti](https://github.com/aleicazatti) in [#102](https://github.com/arviz-devs/preliz/pull/102)
* fix corner case for ylim by [@aloctavodia](https://github.com/aloctavodia) in [#104](https://github.com/arviz-devs/preliz/pull/104)
* change pointinterval style by [@aloctavodia](https://github.com/aloctavodia) in [#105](https://github.com/arviz-devs/preliz/pull/105)
* bump release by [@aloctavodia](https://github.com/aloctavodia) in [#107](https://github.com/arviz-devs/preliz/pull/107)


**Full Changelog**: https://github.com/arviz-devs/preliz/compare/0.0.2...0.0.3

[Changes][0.0.3]


<a id="0.0.2"></a>
# [0.0.2](https://github.com/arviz-devs/preliz/releases/tag/0.0.2) - 2022-11-08



[Changes][0.0.2]


[0.14.0]: https://github.com/arviz-devs/preliz/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/arviz-devs/preliz/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/arviz-devs/preliz/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/arviz-devs/preliz/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/arviz-devs/preliz/compare/0.9.1...0.10.0
[0.9.1]: https://github.com/arviz-devs/preliz/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/arviz-devs/preliz/compare/0.8.1...0.9.0
[0.8.1]: https://github.com/arviz-devs/preliz/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/arviz-devs/preliz/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/arviz-devs/preliz/compare/0.6.3...0.7.0
[0.6.3]: https://github.com/arviz-devs/preliz/compare/0.6.2...0.6.3
[0.6.2]: https://github.com/arviz-devs/preliz/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/arviz-devs/preliz/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/arviz-devs/preliz/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/arviz-devs/preliz/compare/0.4.1...0.5.0
[0.4.1]: https://github.com/arviz-devs/preliz/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/arviz-devs/preliz/compare/0.3.8...0.4.0
[0.3.8]: https://github.com/arviz-devs/preliz/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/arviz-devs/preliz/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/arviz-devs/preliz/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/arviz-devs/preliz/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/arviz-devs/preliz/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/arviz-devs/preliz/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/arviz-devs/preliz/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/arviz-devs/preliz/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/arviz-devs/preliz/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/arviz-devs/preliz/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/arviz-devs/preliz/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/arviz-devs/preliz/compare/0.0.3...0.1.0
[0.0.3]: https://github.com/arviz-devs/preliz/compare/0.0.2...0.0.3
[0.0.2]: https://github.com/arviz-devs/preliz/tree/0.0.2

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.8.1 -->
