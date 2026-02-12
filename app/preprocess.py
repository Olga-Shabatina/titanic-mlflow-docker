def preprocess(data: dict):
    processed = {
        "pclass": data["pclass"],
        "sex": 0 if data["sex"].lower() == "male" else 1,
        "age": data["age"],
        "fare": data["fare"]
    }
    return [list(processed.values())]