MALE = 0
FEMALE = 1

def preprocess(data: dict):
    sex = data["sex"]
    if isinstance(sex, str):
        sex = MALE if sex.lower() == "male" else FEMALE
    return [[data["pclass"], sex, data["age"], data["fare"]]]