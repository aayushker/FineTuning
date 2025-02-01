from transformers import TrainingArguments, Trainer

class fineTuner:
    def __init__(self, model, small_train_dataset, small_eval_dataset, compute_metrics):
        self.model = model
        self.small_train_dataset = small_train_dataset
        self.small_eval_dataset = small_eval_dataset
        self.compute_metrics = compute_metrics

        self.training_args = TrainingArguments(
            output_dir="test_trainer",
            #evaluation_strategy="epoch",
            per_device_train_batch_size=1,  # Reduce batch size here
            per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
            gradient_accumulation_steps=4
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.small_train_dataset,
            eval_dataset=self.small_eval_dataset,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()
