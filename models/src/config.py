class Config:
    ID2LABEL = {
        '공포': 0,
        '놀람': 1,
        '분노': 2,
        '슬픔': 3,
        '중립': 4,
        '행복': 5,
        '혐오': 6
    }

    LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}