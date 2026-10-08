# Graph interface smoke (Phase 2)

> Decision: **go**

## graph_obs

- ok: True
| Setting | N nodes | N edges | node_dim_ok |
|---|---:|---:|:---:|
| star_4 | 4 | 6 | True |
| star_6 | 6 | 10 | True |
| star_8 | 8 | 14 | True |
| mesh_6 | 6 | 12 | True |
| linear_6 | 6 | 10 | True |

## pointer_actions

- ok: True (action_space sizes vary across N: True)
- action_space_n by N: [69, 151, 265]
- ping flat ids by N: [5, 7, 9]

Run: `python -m src.run_graph_interface_smoke`
