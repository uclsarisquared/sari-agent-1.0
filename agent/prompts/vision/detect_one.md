Detect the <target_object> from the provided info about it. The box_2d should be [xmin, ymin, xmax, ymax] in the image normalized to 0-1000. The top-left corner of the image is the origin. The x- and y-axes go horizontally and vertically, respectively. Return one JSON object with a label. If the target is genuinely not visible, return the empty JSON array []. Never return masks or code fencing. Limit to one object only. Example output:

```json
{'box_2d': box_2d, 'label': target_object}
```

