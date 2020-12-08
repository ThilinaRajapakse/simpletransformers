The [Weights & Biases](https://www.wandb.com/) framework is supported for visualizing model training.

To use this, simply set a project name for W&B in the `wandb_project` attribute of the `args` dictionary. This will log all hyperparameter values, training losses, and evaluation metrics to the given project.

```python
model = ClassificationModel('roberta', 'roberta-base', args={'wandb_project': 'project-name'})
```

For a complete example, see [here](https://medium.com/skilai/to-see-is-to-believe-visualizing-the-training-of-machine-learning-models-664ef3fe4f49).