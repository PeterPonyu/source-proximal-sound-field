# Registration record

Every confirmatory study was registered by a commit to the project repository before its first output, together with the SHA-256 of every frozen input (`archive/2026-11-05-taslp/acoustic/FROZEN-INPUTS.json`). Post hoc analyses had a plan committed before their script, with three exceptions: the plans of the spherical-array fitting analysis, the exact ball predictor and the run times of the point baselines (`posthoc/upgrade-wp1`, `upgrade-wp2`, `upgrade-timing`) were written before their code but were not committed before it, so they are not listed. Each run records the registration commit and its start time in `started.json` next to its results.

The table lists every registered document: the time of the commit that added it (UTC), the commit in the project repository, and the SHA-256 of the document as committed. The documents themselves are not public; these checksums identify them.

Paths are relative to `archive/2026-11-05-taslp/acoustic/`. The two studies of the earlier single-frequency estimator (`TB-S1`, `TB-M1`) informed its development and supply no number of the paper; their results are not released.

| Committed (UTC) | Document | Commit | SHA-256 as committed | Changed since |
|---|---|---|---|---|
| 2026-11-05 04:07 | `PROTOCOLS/TB-S1.md` | `c42efd3` | `c4621136cd0d1263c767d767e55b00d8754db9fc761f85a680afe9a643020767` | no |
| 2026-11-05 04:42 | `PROTOCOLS/TB-S1-AMENDMENT-1.md` | `e782232` | `01ac0495e5bd07d15ce3144cd6d9735e93fab5772f04b2bf731997ab51262951` | no |
| 2026-11-05 05:20 | `PROTOCOLS/TB-M1.md` | `4eda109` | `1d7e81e21b8e2be2ded6ea69c0924913bf167ef93fc3cc36dcfe52cb3b26dc54` | no |
| 2026-11-05 07:33 | `V2-DESIGN.md` | `035557d` | `bbadde6f0afb482fdfa6f94e5b43b2273f3acb04a10d5b86972b98fea9b1e367` | no |
| 2026-11-05 10:39 | `V2-DESIGN-AMENDMENT-1.md` | `3ba6c72` | `49364481d1a6805aa8bcbf7bfc6df468e50520bfeb4b78fb750af970deed326a` | no |
| 2026-11-05 10:50 | `PROTOCOLS/TB-M2.md` | `f4d1e18` | `7099c8da0e5674e0f72d34f6d43004ac024f846e8e8fc92ed3536cab9fccb193` | no |
| 2026-11-05 10:50 | `PROTOCOLS/TB-S2.md` | `f4d1e18` | `6669e88718658572478e1066ee5a73edb0772ac580b1b40416995a1d6bb9a111` | no |
| 2026-11-05 12:56 | `PROTOCOLS/TB-M2-RIB.md` | `bdd786b` | `bacca6c9dd46448aa23a3d43431d8ed442bd9365c80b2546657eb7c3f58680db` | no |
| 2026-11-05 12:56 | `PROTOCOLS/TB-M2-SBLFIX.md` | `bdd786b` | `91c4c5edad9e8b1970361629b93f5ec87d914b9414386189b7e9b9f2333a69d3` | no |
| 2026-11-05 13:02 | `PROTOCOLS/TB-M2-RIB-AMENDMENT-1.md` | `9904d8a` | `651b30e4f2036573c88f3741fdeb2eb719f52b8c174a8e6eec39f1c3969d8525` | no |
| 2026-11-06 00:16 | `PROTOCOLS/TB-M2-RIB-AMENDMENT-2.md` | `70cf8aa` | `a65248f11524c805e76b92038178d6edcf3178bfbe6388240bf4aaafec21addb` | no |
| 2026-11-06 07:11 | `PROTOCOLS/TB-PROX.md` | `196f5f9` | `8ac5767c13effb6e1656c96abc8286a8f5bfdfc02309a0b7610608b60144a472` | no |
| 2026-11-06 07:11 | `PROX-DESIGN.md` | `196f5f9` | `a2f398978625c8a52c7b4996dc8315f4b3378103df582a74fb0f4b9bb133e60d` | no |
| 2026-11-06 10:02 | `PROTOCOLS/TB-M2-EXACT.md` | `c493837` | `c843499f3e7e39dbe36f8098f2e43e5c23c5485068e95e58ab89791d5ae6a4c0` | no |
| 2026-11-06 10:42 | `PROTOCOLS/TB-PROX-AMENDMENT-1.md` | `02b4543` | `b2646bff41a61fc262cb841166686ab78e6b2ca04fa515a85ec7d579df1b5462` | no |
| 2026-11-06 10:42 | `PROTOCOLS/TB-PROX-S2.md` | `bf0b011` | `b03cff8af39b63f6c42fa997d5c5c15b6272566aeeb129a62634cc7ce796c3f6` | no |
| 2026-11-06 11:42 | `PROTOCOLS/TB-M2-EXACT-HI.md` | `8e17469` | `bde46b74829b08ae194371c8f2970b6a8c93475f86c0cb2ad7b5c468fa493b2f` | no |
| 2026-11-07 01:27 | `posthoc/crb/PLAN.md` | `699cc21` | `ed78d2102fb8c341b4a4d9a9ee38e67528f41e0e15ee01cb8dc98368dabb836f` | no |
| 2026-11-07 01:45 | `studies/AUDITS/kernel-envelope-2026-11-06/PLAN.md` | `47415c6` | `3e46122491a4db322a16fdc6bd731828c61bdee399e796848cc6c1a399db6337` | no |
| 2026-11-07 06:38 | `posthoc/fixed-order/PLAN.md` | `86272cb` | `6c0ab95780b9340b5fb9fc38b17f99369f9ad41748623049f9e7473dc6e50238` | yes |
| 2026-11-07 08:33 | `posthoc/margins/PLAN.md` | `3a9bb63` | `f10d97687ad1e1bb2ea2c252b428cd0bbb309b17800a2c026efdfe3af9fb838d` | no |
| 2026-11-07 11:45 | `posthoc/review3/PLAN.md` | `7c23306` | `b85d73621f77ba0f0fc8e6caa51cc373a189078f7e23bd237387d2af171bc2de` | no |
| 2026-11-07 11:58 | `posthoc/review3/PLAN-C-REPRODUCTION.md` | `6417915` | `6e6f7469585c39bd0408a279b85ebd7e6517cbc71bdba5f275b58e879c996e35` | no |
| 2026-11-07 12:43 | `posthoc/review3/PLAN-D-GAUSSIAN-EXTENSION.md` | `41b957d` | `e34262d8e6a62de6cc0ee5e6d3b1671767446a9837e5ea8ba9569f501f9fe7fe` | no |
| 2026-11-07 12:43 | `posthoc/review3/PLAN-E-GROUP-SPARSE-REPRODUCTION.md` | `41b957d` | `d742e7558a495d2687127b4da9a9cba348190c32f90b060b7526173587f87b94` | no |
| 2026-11-07 12:54 | `posthoc/review3/PLAN-C2-DIAGNOSTIC.md` | `d3717eb` | `48afbceea5de4d8ea1eae5486c89ff9280cb5163669a75461cfc922a37f1f209` | no |
| 2026-11-07 12:59 | `posthoc/review3/PLAN-F-CAP8.md` | `f906c35` | `6b8d41576d653ade4978fae9a199170c289f049fa0eb45a9a1e5459218f92798` | no |
| 2026-11-07 13:08 | `studies/TB-M3-MIRACLE/PROTOCOL.md` | `916cf1e` | `3c3037b856bbc56f006fb32bb264a2f35c4feb41aa20939072b85e5949ce1889` | no |
| 2026-11-09 06:35 | `posthoc/review4/PLAN.md` | `f78265c` | `52a27e3dcc25ba18c2223347a50b80b4eac581c2ef9b8c27aa196751e1eace67` | no |
| 2026-11-10 03:02 | `posthoc/upgrade-revfit/PLAN.md` | `0a89ff0` | `0a26b5629e9f135893e70d0f9a8384343fad7bd10517a4b78c6c56a6ca66a137` | no |
| 2026-11-10 03:02 | `upgrade-v3/PLAN-3b.md` | `0a89ff0` | `974379a3471039c435839229202e5c5dbd4b94afac5ba528911651ebcb07cfb6` | no |
| 2026-11-10 03:02 | `upgrade-v3/PLAN.md` | `0a89ff0` | `279f6f2aacc63e5f4af4842e0a9e96d3e7307b909f641c6d1f0f05c1b709739f` | no |
| 2026-11-10 03:03 | `posthoc/upgrade-mfsbl/PLAN.md` | `76f2e2a` | `4a129c6420fed4e2fb2987e318cc1d1ac27bf0fbf7bb94c129cee89fca16ead5` | no |
| 2026-11-10 03:03 | `studies/TB-M4-MIRACLE-A2/PROTOCOL.md` | `76f2e2a` | `d4fedc27ee9be337f476d0c5828a4cf853193001c3b2a761a8849ee99b2edd48` | no |
| 2026-11-10 13:51 | `posthoc/upgrade-timing/PLAN.md` | `1b571e7` | `577841232929526e89bc6f53dbbed6dcdbd73c5d776e843e949acb76b333590a` | no |
| 2026-11-10 13:51 | `posthoc/upgrade-wp1/PLAN.md` | `1b571e7` | `f9a1b6e3ba4dc9472f5889d31b292b4a6dfc7c96cda15be537f3a7156520ee4d` | no |
| 2026-11-10 13:51 | `posthoc/upgrade-wp2/PLAN.md` | `1b571e7` | `967cc0a448aa01b593485c422d7108171721e12405ce64568b4c943ab549570b` | no |
