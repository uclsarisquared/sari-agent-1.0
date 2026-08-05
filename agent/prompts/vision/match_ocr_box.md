An OCR tool was used to extract texts from the image. Find the most semantically similar bounding box to the <target_object>. An Embodied AI Agent will be using this bounding box to center the agent's perspective on the target. You will receive a list of bounding boxes and their labels along with the <target_object>. Return the bounding box that best matches the <target_object>. Example output:

```json
{'box_2d': box_2d, 'label': target_object}
```

