# Full Baseline + D Injection Comparison

| task_type | baseline | injected | Δ success | Δ rate | invalid rate | parse rate | repairs | paired F->S | paired S->F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 91/134 | 103/134 | 12 | 8.955 pp | 0.304770 -> 0.284511 | 0.066948 -> 0.044787 | 2 | 13 | 1 |
| pick_and_place_simple | 22/24 | 22/24 | 0 | 0.000 pp | 0.103261 -> 0.103261 | 0.036364 -> 0.017699 | 0 | 0 | 0 |
| pick_clean_then_place_in_recep | 21/31 | 20/31 | -1 | -3.226 pp | 0.428325 -> 0.444062 | 0.053908 -> 0.041775 | 0 | 0 | 1 |
| pick_heat_then_place_in_recep | 20/23 | 20/23 | 0 | 0.000 pp | 0.184466 -> 0.184818 | 0.040268 -> 0.019277 | 0 | 0 | 0 |
| pick_cool_then_place_in_recep | 18/21 | 18/21 | 0 | 0.000 pp | 0.497462 -> 0.478261 | 0.000000 -> 0.000000 | 0 | 0 | 0 |
| look_at_obj_in_light | 2/18 | 14/18 | 12 | 66.667 pp | 0.366071 -> 0.238636 | 0.110667 -> 0.042553 | 2 | 12 | 0 |
| pick_two_obj_and_place | 8/17 | 9/17 | 1 | 5.882 pp | 0.100719 -> 0.101449 | 0.140086 -> 0.149184 | 0 | 1 | 0 |
