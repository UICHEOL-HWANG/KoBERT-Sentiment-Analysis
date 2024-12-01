from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TrainingManager:

    @staticmethod
    def compute_metrics(pred):
        """

        :param pred: Tariner에서 반환된 모델
        :return: 평가지표
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    @staticmethod
    def configure_training(output_dir, num_train_epochs=5, learning_rate=2e-5, batch_size=32):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=3,
            fp16=True
        )


    @staticmethod
    def train_model(model,train_dataset, eval_dataset, training_args, early_stopping_patience=2):
        """

        :param model: 설계할 모델명
        :param training_args: 아규먼트
        :param train_dataset: 훈련세트
        :param eval_dataset: 검증세트
        :return: model
        """
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=TrainingManager.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )
            # 학습 시작
            print(f'학습시작: 경로 -> {training_args.output_dir}')
            trainer.train()

            if eval_dataset:
                metrics = trainer.evaluate()
                print(f'검증결과 {metrics}')
            save_path = os.path.join(training_args.output_dir, 'KoBERT-Sentiment-Analysis')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save_model(save_path)
            print(f"{save_path} 경로로 모델 저장 완료")
        except Exception as e:
            print(f'훈련 중 오류 발생 {e}')