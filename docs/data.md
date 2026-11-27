# Data sources, licences and caveats

Where every dataset comes from, how fresh it is, and what it does not cover.

[← Back to the README](../README.md)

* **Events.** [StatsBomb open data](https://github.com/statsbomb/open-data). *Data provided by StatsBomb.*
  CC BY-NC-SA 4.0: attribution required, non-commercial use. A commercial club deployment would need a licensed
  event feed (StatsBomb, Opta, Wyscout) through the same ingest interface.
* **Matches.** [xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)
  (football-data.co.uk + clubelo), updated through 2026-09-03.
* **Transfermarkt.** [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), a public
  snapshot to 6 July 2026 that its author has stopped refreshing. Valuations are Transfermarkt's crowd-sourced
  estimates; fees are as recorded there. Squad values derived from it are frozen at that snapshot: 2026-27 club
  and international forecasts use pre-summer-2026 squads, and clubs promoted in 2026 have no value.
* **Players.** [305kishan/FIFA](https://github.com/305kishan/FIFA): sofifa scrapes of EA FIFA 15–22. FIFA 22 is the
  newest player edition in the public sources this pipeline uses, so player outputs describe that snapshot plus one
  projected season. A newer sofifa export with the same schema can be dropped into `data/raw/`.
* Football is noisy. ~50% 1X2 accuracy is what well-calibrated public-information models achieve, and the betting
  market is the benchmark to beat. Any model claiming 70–90% pre-match accuracy is leaking the result.
