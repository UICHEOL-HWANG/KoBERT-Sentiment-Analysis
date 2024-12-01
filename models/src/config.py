class Config:
    LABEL2ID = {
        '공포': 0,
        '놀람': 1,
        '분노': 2,
        '슬픔': 3,
        '중립': 4,
        '행복': 5,
        '혐오': 6
    }

    ID2LABEL = {v: k for k, v in LABEL2ID.items()}