识别 3x3 九宫格中所有正确格子。
坐标使用 0 基 `[row,col]`，范围均为 `0..2`。
只返回符合 `ImageBinaryChallenge` 的 JSON：保留 `challenge_prompt`，`coordinates` 中每项格式为 `{"box_2d":[row,col]}`。
