import argparse
import torch

from src.dataset_manager import DatasetLoader
from src.model_manager import ModelManager
from src.trainning_manager import TrainingManager
from src.config import Config

def parser_args():
    parser = argparse.ArgumentParser(description='모델 파라미터 튜닝')

    # Data paths
    parser.add_argument('--train_path', type=str, default='../data/raw/train_dataset.json',
                        help='Path to the training dataset JSON file')
    parser.add_argument('--val_path', type=str, default='../data/raw/val_dataset.json',
                        help='Path to the validation dataset JSON file')

    # Model parameters
    parser.add_argument('--model_save_path', type=str, default='models/bert_emotion_model',
                        help='Directory to save the trained model')

    # Hyperparameters
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')

    # Other settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train the model on (cuda or cpu)')
    parser.add_argument('--quantize', action='store_true',
                        help='Enable dynamic quantization for the model')

    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    num_labels = len(Config.ID2LABEL)

    # CUDA 설정 확인
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA가 사용 불가능합니다. CPU로 실행됩니다.")
        args.device = 'cpu'
    else:
        print(f"사용할 디바이스: {args.device}")

    model_manager = ModelManager(base_model='monologg/kobert', device=args.device, quantize=args.quantize)
    model, tokenizer = model_manager._load_model_and_tokenizer(num_labels=num_labels)

    dataset_loader = DatasetLoader(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    datasets = dataset_loader.get_datasets()
    train = datasets.get('train')
    valid = datasets.get('val')

    training_args = TrainingManager.configure_training(
        output_dir=args.model_save_path,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    TrainingManager.train_model(model, train, valid, training_args)

if __name__ == '__main__':
    main()



