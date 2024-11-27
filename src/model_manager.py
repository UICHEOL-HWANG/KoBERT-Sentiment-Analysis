from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import os

class ModelManager:
    def __init__(self, base_model="monologg/kobert", device="cpu", quantize: bool = False):
        """
        모델 초기화
        :param base_model:
        """
        self.base_model = base_model
        self.token = self._load_environment()
        self.device = device
        self.quantize = quantize

    @staticmethod
    def _load_environment():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(current_dir, '.env')
        load_dotenv(dotenv_path)

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("Hugging Face 토큰을 .env 파일에 설정해주세요. 예: HUGGINGFACE_TOKEN=your_token")
        return token

    def _login_huggingface(self):
        login(token=self.token)
        print('허깅페이스 로그인 완 ')

    def _load_model_and_tokenizer(self, num_labels):
        model = AutoModel.from_pretrained(self.base_model, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)

        tokenizer.padding_side = "right"  # 패딩을 오른쪽에 적용
        if tokenizer.pad_token is None:
            # PAD 토큰이 정의되어 있지 않은 경우, EOS 토큰을 PAD 토큰으로 설정
            tokenizer.pad_token = tokenizer.eos_token
            print(f"PAD 토큰이 정의되어 있지 않아 EOS 토큰을 PAD 토큰으로 설정했습니다: {tokenizer.pad_token}")
        model.to(self.device)

        if self.quantize:
            print('모델 동적 양자화 ..')
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print('모델 동적 양자화 완료')
        # 모델을 지정된 장치로 이동
        print(f'{self.base_model} 모델과 토크나이저가 성공적으로 로드되었습니다.')
        return model, tokenizer



