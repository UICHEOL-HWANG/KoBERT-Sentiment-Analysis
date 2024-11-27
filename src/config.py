class Config:
    ID2LABEL = {
        0: '분노',    # angry
        1: '행복',    # happy
        2: '슬픔',    # sad
        3: '공포',    # fear
        4: '혐오',    # disgust
        5: '놀람',    # surprise
        6: '중립'     # neutral
    }

    LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}