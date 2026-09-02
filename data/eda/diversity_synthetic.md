# SFT diversity report

40 record(s), 40 unique instruction(s).

## Instruction phrasing

| metric | value | reading |
| --- | --- | --- |
| distinct-1 | 0.433 | unique unigrams / total |
| distinct-2 | 0.583 | lower = more repetition |
| distinct-3 | 0.700 |  |
| distinct-4 | 0.873 |  |
| nearest-neighbour similarity (mean) | 0.515 | higher = more templated |
| near-template rate | 50% | share with a near-twin |
| shared opening 3-gram | 50% | 'اشرح معنى كلمة' |

## Region spread

normalised entropy **0.71** (1.00 = uniform), max share 60% in 'najdi'

| region | records | share |
| --- | --- | --- |
| najdi | 24 | 60% |
| southern | 8 | 20% |
| northern | 4 | 10% |
| eastern | 3 | 8% |
| western | 1 | 2% |

## Format spread

normalised entropy **0.35** (1.00 = uniform), max share 80% in 'dictionary_entry'

| format | records | share |
| --- | --- | --- |
| dictionary_entry | 32 | 80% |
| prose | 5 | 12% |
| narrative_paragraph | 0 | 0% |
| verse | 0 | 0% |
| footnote_block | 0 | 0% |
| list | 3 | 8% |

## Flags

- TEMPLATED SUBSET: 50% of instructions have a near-twin (>= 25%)
- SHARED OPENING: 50% of instructions start with 'اشرح معنى كلمة' (>= 35%)
- REGION IMBALANCE: normalised entropy 0.71 < 0.75 ('najdi' holds 60%)
- FORMAT IMBALANCE: normalised entropy 0.35 < 0.75 ('dictionary_entry' holds 80%)

_Bands are provisional and derived from a synthetic set; real data should
set them. A region warning may reflect the corpus's own known imbalance
(`western` is under-resourced at 48% of the median) rather than a sampling
defect._
