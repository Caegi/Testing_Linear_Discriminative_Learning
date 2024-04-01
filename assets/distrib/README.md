This is Flexique version 1.3.2 from Feb, 2 2024

Flexique derives from Lexique: http://www.lexique.org/

Flexique is distributed under a Creative Commons Attribution-NonCommercial-ShareAlike licence, see http://creativecommons.org/licenses/by-nc/4.0/

Documentation is available at:

http://www.llf.cnrs.fr/flexique-fr.php

If you use Flexique and wish to refer to it, please reference the following paper:

Olivier Bonami, Gauthier Caron et Clément Plancq. 2014. Construction d'un lexique flexionnel phonétisé libre du français. Actes du quatrième Congrès Mondial de Linguistique Française, pp. 2583--2596. 
http://www.shs-conferences.org/articles/shsconf/pdf/2014/05/shsconf_cmlf14_01223.pdf

# [v1.3.2] - 08 Feb 2024

- Corrects variants in a few verbs
- Adds variant for dîner (diner) and traîner (trainer).
- Adds téléporter

# [v1.3] - 27 Nov 2023

## Fixed

### Verbs

- verbs in -eter which were missing the final vowel: "breveter", "colleter", "fureter", "caqueter", "cliqueter", "déchiqueter", "empaqueter", "étiqueter", "feuilleter", "hoqueter", "piqueter", "refeuilleter", "becqueter"


# [v1.2] - 23 Nov 2023

Changes from version 1.1, July 30th 2014. Numerous changes in the verbs file were provided by Sacha Beniamine.

## Added

- License pdf

### Nouns

- 'mort', 'voix' (this was done in 2016 and already reflected in distributed archives)

### Verbs

- Add missing frequent verbs, among which: 'craindre', 'défendre', 'dénuer', 'dépourvoir', 'geindre', 'gîter', 'hanter', 'nuire', 'parer', 'parfaire', 'parvenir', 'passer', 'payer', 'rendre', 'saboter', 'saisir', 'saper', 'sauter', 'sembler', 'semer', 'sentir', 'servir', 'siffler', 'signer', 'songer', 'souder', 'souvenir', 'structurer', 'surgir', 'suturer', 'tarer'. Where relevant, removed them as variants of other verbs.
- Added 'poser', 'rassir'  (this was done in 2016 and already reflected in distributed archives)
- Added numerous missing verbs starting in s-, t- and v- (this was done in 2016 and already reflected in distributed archives)
- Overabundance: verbs in -ayer (eg. paie/paye), derivatives from 'faire' (faîtes/faisez), 'dire' (dîtes/disez), 'haïr' (/ai/ vs /E/)
  - Overabundant forms are separated by ";"

## Fixed

### Nouns

- Fixed nouns in -Cment to -Cəment, eg. /akjEsmɑ/ to /akjEsəmɑ̃/ (this was done in 2016 and already reflected in distributed archives)

### Verbs

- Changed "oe" in 'ameuter', 'rameuter' from œ to ø and Ø
- Corrected defectives prefixed in re-
- Separated affermir/rassir
- Removed final -ʁ from the infinitive of 'aller', 'envoyer', 'renvoyer'
- Restored missing eks- at the start of words in ex-  (this was done in 2016 and already reflected in distributed archives)
- Restored ʒ instead of əs at the start of words in g- and j- ('geler', 'jardiner', etc)
- Restored ʒ instead of s at the start of 'jaboter', 'jalonner', jalouser, japoniser, etc
- Removed duplicate rows for 'gréer', 'iodler', 'relâcher'
- Corrected to words in pâ-, where sy- was present instead of pa-
- Corrected lemmas where pan- was present instead of syn- (eg. 'panchroniser' -> 'synchroniser')
- Sorted rows
- Corrected defective entry for 'revaloir'
- Corrected typo 'difracter' -> 'diffracter'

## Changed

### Verbs

- Removed numerous variants in re-, introduced them as separate verbs  (this was done in 2016 and already reflected in distributed archives)
- Removed dots between person and number, which are now considered a single feature, eg. prs.1.sg to prs.1sg
- Merged duplicate paradigms assener/asséner, co-habiter/cohabiter,
    exhausser/exaucer, pommer/paumer, roder/rôder, zieuter/zyeuter
- Separated bailler/bayer, rechampir/réchampir, enter/hanter 
- Merged 'haïr2' with 'haïr', using overabundance (they are the same verb, with the same meaning)
