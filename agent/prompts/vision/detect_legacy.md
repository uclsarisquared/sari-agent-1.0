Detect the {{TARGET_NAME}} in the image. The box_2d should be [xmin, ymin, xmax, ymax] in the image normalized to 0-1000. The top left corner of the image is the origin. The x and y axis go horizontally and vertically, respectively. Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 1 object only. Here is an example output:

```json
{'box_2d': box_2d, 'label': label_name}
```
