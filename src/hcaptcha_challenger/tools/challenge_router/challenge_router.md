识别图片中的 `challenge_prompt` 与 `challenge_type`。
`challenge_type` 只能是：`image_label_single_select`、`image_label_multi_select`、`image_drag_single`、`image_drag_multi`。
点击/选择类任务用 `image_label_*`；拖拽/放置/排列类任务用 `image_drag_*`。
只要求一个目标用 `*_single`；要求多个目标、两项以上或 9 宫格选择用 `*_multi`。
只返回符合 `ChallengeRouterResult` 的 JSON。
