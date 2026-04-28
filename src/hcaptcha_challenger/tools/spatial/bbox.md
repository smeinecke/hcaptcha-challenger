结合图片与挑战提示，只关注主挑战画布。
找到满足提示要求的目标区域，并给出覆盖该目标的最小整数像素边界框。
只返回符合 `ImageBboxChallenge` 的 JSON：在 `challenge_prompt` 中回填原始提示，并使用 `bounding_boxes` 字段输出框坐标。
